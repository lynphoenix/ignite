from pathlib import Path
from datetime import datetime
import os

import fire
import time

import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim

import ignite
import ignite.distributed as idist
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import manual_seed, setup_logger

from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear

import utils


def training(local_rank, config):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name="ImageNet-Training", distributed_rank=local_rank)

    log_basic_info(logger, config)

    output_path = config["output_path"]
    if rank == 0:
        if config["stop_iteration"] is None:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            now = "stop-on-{}".format(config["stop_iteration"])

        folder_name = "{}_backend-{}-{}_{}".format(config["model"], idist.backend(), idist.get_world_size(), now)
        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_path"] = output_path.as_posix()
        logger.info("Output path: {}".format(config["output_path"]))

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

    # Setup dataflow, model, optimizer, criterion
    train_loader, test_loader = get_imagenet_dataloader(config)

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    # Create trainer for current task
    trainer = create_supervised_trainer(model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger)

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion),
    }

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED, run_validation)

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

    # Store 3 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=get_save_handler(config),
        evaluator=evaluator,
        models={"model": model},
        metric_name="accuracy",
        n_saved=3,
        trainer=trainer,
        tag="test",
    )

    # In order to check training resuming we can stop training on a given iteration
    if config["stop_iteration"] is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config["stop_iteration"]))
        def _():
            logger.info("Stop training on {} iteration".format(trainer.state.iteration))
            trainer.terminate()

    @trainer.on(Events.ITERATION_COMPLETED(every=20))
    def print_acc(engine):
        if rank == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"\
                    .format(engine.state.epoch, engine.state.iteration, len(train_loader),
                            engine.state.saved_batch_loss
                            ))

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    if rank == 0:
        tb_logger.close()


def run(
    seed=543,
    data_path="/data/ImageNet/ImageNet-pytorch/",
    output_path="/data/ImageNet/ImageNet-pytorch/output/",
    model="resnet18",
    batch_size=512,
    momentum=0.9,
    weight_decay=1e-4,
    num_workers=16,
    num_epochs=100,
    learning_rate=0.1,
    num_warmup_epochs=4,
    validate_every=3,
    checkpoint_every=200,
    backend="nccl",
    resume_from=None,
    log_every_iters=15,
    nproc_per_node=None,
    stop_iteration=None,
    with_trains=False,
    cache_dataset=True,
    num_classes=1000,
    lr_step_size=30,
    lr_gamma=0.1,
    **spawn_kwargs
):
    """Main entry to train an model on CIFAR10 dataset.

    Args:
        seed (int): random state seed to set. Default, 543.
        data_path (str): input dataset path. Default, "/tmp/cifar10".
        output_path (str): output path. Default, "/tmp/output-cifar10".
        model (str): model name (from torchvision) to setup model to train. Default, "resnet18".
        batch_size (int): total batch size. Default, 512.
        momentum (float): optimizer's momentum. Default, 0.9.
        weight_decay (float): weight decay. Default, 1e-4.
        num_workers (int): number of workers in the data loader. Default, 12.
        num_epochs (int): number of epochs to train the model. Default, 24.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 0.4.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay. Default, 4.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 200.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        log_every_iters (int): argument to log batch loss every ``log_every_iters`` iterations.
            It can be 0 to disable it. Default, 15.
        stop_iteration (int, optional): iteration to stop the training. Can be used to check resume from checkpoint.
        with_trains (bool): if True, experiment Trains logger is setup. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    print(config)

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:

        parallel.run(training, config)


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    train_dataset, test_dataset = utils.get_train_test_datasets(config["data_path"])

    if idist.get_rank() == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"],
        num_workers=config["num_workers"], shuffle=True, 
        drop_last=True,
    )

    test_loader = idist.auto_dataloader(
        test_dataset, batch_size=2 * config["batch_size"], 
        num_workers=config["num_workers"], shuffle=False,
    )
    return train_loader, test_loader


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    print("Loading validation data")
    test_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    '''
    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    '''

    return train_dataset, test_dataset


def get_imagenet_dataloader(config):
    train_dir = os.path.join(config["data_path"], 'train')
    val_dir = os.path.join(config["data_path"], 'val')
    print(train_dir)
    print(val_dir)
    train_dataset, test_dataset = load_data(train_dir, val_dir,
                            config["cache_dataset"])
    print(len(train_dataset))
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"],shuffle=True,
        num_workers=config["num_workers"], pin_memory=True)

    test_loader = idist.auto_dataloader(
        test_dataset, batch_size=config["batch_size"],shuffle=False,
        num_workers=config["num_workers"], pin_memory=True)

    return train_loader, test_loader


def initialize(config):
    model = utils.get_model(config["model"], config["num_classes"])
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
#        nesterov=True,
    )
    optimizer = idist.auto_optim(optimizer)

    # criterion = nn.CrossEntropyLoss().to(idist.device())
    criterion = nn.CrossEntropyLoss()

    le = config["num_iters_per_epoch"]
    cl = config["learning_rate"]

    milestones_values = [
        (30*le, cl),
        (50*le, 0.5*cl),
        (46*le, 0.1*cl),
        (60*le, 0.1*cl),
        (61*le, 0.01*cl),
        (90*le, 0.01*cl),
        (100*le, 0.01*cl),
        (120*le, 0.001*cl),
        # (le * config["num_warmup_epochs"], config["learning_rate"]),
        # (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    logger.info(
        "\nEpoch {} - elapsed: {} - {} metrics:\n {}".format(
            epoch, elapsed, tag, "\n".join(["\t{}: {}".format(k, v) for k, v in metrics.items()])
        )
    )


def log_basic_info(logger, config):
    logger.info("Train {} on CIFAR10".format(config["model"]))
    logger.info("- PyTorch version: {}".format(torch.__version__))
    logger.info("- Ignite version: {}".format(ignite.__version__))

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info("\t{}: {}".format(key, value))
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info("\tbackend: {}".format(idist.backend()))
        logger.info("\tworld size: {}".format(idist.get_world_size()))
        logger.info("\n")


def create_supervised_trainer(model, 
                              optimizer, 
                              criterion, 
                              lr_scheduler, 
                              train_sampler, 
                              config, 
                              logger):
    device = idist.device()

    def _update(engine, batch):

        model.train()

        # x, y = batch[0], batch[1]
        (imgs, targets) = batch


        # if imgs.device != device:
        #    imgs = imgs.to(device, non_blocking=True)
        #    target = target.to(device, non_blocking=True)

        # model.train()
        # (imgs, targets) = batch
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # targets = [target.to(device, non_blocking=True) for target in targets
        #            ]  #if torch.cuda.device_count() >= 1 else targets

        outputs = model(imgs)
        # print(outputs.shape)
        # print(targets.shape)
        loss = criterion(outputs, targets)

        # dist_metrics = [reduce_metric_dict(me) for me in _metrics]

        # Compute gradient
        optimizer.zero_grad()
        # loss = sum(total_loss)
        loss.backward()
        optimizer.step()

        # This can be helpful for XLA to avoid performance slow down if fetch loss.item() every iteration
        acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        if config["log_every_iters"] > 0 and (engine.state.iteration - 1) % config["log_every_iters"] == 0:
            batch_loss = loss.item()
            engine.state.saved_batch_loss = batch_loss
        else:
            batch_loss = engine.state.saved_batch_loss
        '''
        if idist.get_rank() == 0:
            print(acc1)
            print(acc5)
            print(batch_loss)
        '''
        return {
            "batch loss": batch_loss,
        }

    trainer = Engine(_update)
    trainer.state.saved_batch_loss = -1.0
    trainer.state_dict_user_keys.append("saved_batch_loss")
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_trainer(model, 
                   optimizer, 
                   criterion, 
                   lr_scheduler, 
                   train_sampler, 
                   config, 
                   logger):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    def train_step(engine, batch):

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()
        # Supervised part
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # This can be helpful for XLA to avoid performance slow down if fetch loss.item() every iteration
        if config["log_every_iters"] > 0 and (engine.state.iteration - 1) % config["log_every_iters"] == 0:
            batch_loss = loss.item()
            engine.state.saved_batch_loss = batch_loss
        else:
            batch_loss = engine.state.saved_batch_loss

        return {
            "batch loss": batch_loss,
        }

    trainer = Engine(train_step)
    trainer.state.saved_batch_loss = -1.0
    trainer.state_dict_user_keys.append("saved_batch_loss")
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def get_save_handler(config):
    if config["with_trains"]:
        from ignite.contrib.handlers.trains_logger import TrainsSaver

        return TrainsSaver(dirname=config["output_path"])

    return DiskSaver(config["output_path"], require_empty=False)


if __name__ == "__main__":
    fire.Fire({"run": run})
