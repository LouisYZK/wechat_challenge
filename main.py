import logging
import os
import time
import torch

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import mlflow
from mlflow import log_metric, log_param, log_artifacts


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        args.n_gpu = torch.cuda.device_count()
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    if args.device == 'cpu':
        device = torch.device("cpu")
        # model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        model = DDP(model)

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    if args.device == 'cuda':
        with mlflow.start_run(run_name=args.exp_name):
            for epoch in range(args.max_epochs):
                train_loss = .0
                epoch_step = 0
                for batch in train_dataloader:
                    model.train()
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    step += 1
                    epoch_step += 1
                    if step % args.print_steps == 0:
                        time_per_step = (time.time() - start_time) / max(1, step)
                        remaining_time = time_per_step * (num_total_steps - step)
                        remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                        logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
                    train_loss += loss
                train_loss /= epoch_step
                mlflow.log_metric(key='loss', value=f'{train_loss:.3f}', step=epoch)
                # 4. validation
                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                for k, v in results.items():
                    mlflow.log_metric(key=k, value=v, step=epoch)

                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                            f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
            mlflow.log_metric(key='best_score', value=best_score)

    elif  args.device == 'cpu' and dist.get_rank() == 0:
        with mlflow.start_run(run_name=f'{args.exp_name}'):
            for epoch in range(args.max_epochs):
                train_dataloader.sampler.set_epoch(epoch)
                for batch in train_dataloader:
                    model.train()
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    step += 1
                    if step % args.print_steps == 0:
                        time_per_step = (time.time() - start_time) / max(1, step)
                        remaining_time = time_per_step * (num_total_steps - step)
                        remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                        logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

                # 4. validation
                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                mlflow.log_metric(key='loss', value=loss, step=epoch)
                for k, v in results.items():
                    mlflow.log_metric(key=k, value=v, step=epoch)

                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                            f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}_{args.local_rank}.bin')
    elif args.device == 'cpu' and dist.get_rank() > 0 :
        for epoch in range(args.max_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            for batch in train_dataloader:
                model.train()
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                step += 1
                if step % args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, step)
                    remaining_time = time_per_step * (num_total_steps - step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            loss, results = validate(model, val_dataloader)
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
    
    else:
        raise NotImplementedError('No such device type!')
def main():
    args = parse_args()
    setup_logging()
    # setup_device(args)
    setup_seed(args)
    if args.device == 'cpu':
        dist.init_process_group(backend='gloo')

    args.savedmodel_path = f'save/{args.exp_name}'
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
