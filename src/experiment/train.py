import json
import argparse

from src.experiment import common_functions as cmf
from src.utils import timer
from src.utils.comm import synchronize, get_rank

import torch

""" Get parameters """
def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="src/experiment/options/default.yml", help="Path to config file.")
	parser.add_argument("--method_type",
        default="ensemble", help="Method type among [||].")
	parser.add_argument("--dataset",
        default="didemo", help="Dataset to train models [|].")
	parser.add_argument("--num_workers", type=int,
        default=4, help="The number of workers for data loader.")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")
        
	parser.add_argument("--distributed", action='store_true')
	parser.add_argument("--local_rank", type=int, default=0)

	params = vars(parser.parse_args())
	if params['local_rank'] == 0:
		print(json.dumps(params, indent=4))
	return params

""" Training the network """
def train(config, local_rank=0, distributed=False):

    # create loggers
    it_logger = cmf.create_logger(config, "ITER", "train.log") if local_rank==0 else None
    eval_logger = cmf.create_logger(config, "EPOCH", "scores.log") if local_rank==0 else None

    """ Prepare data loader and model"""
    dsets, L = cmf.get_loader(dataset, split=["train", "test", "phrase"],
                              loader_configs=[config["train_loader"], config["test_loader"], config["phrase_loader"]],
                              num_workers=config["misc"]["num_workers"], distributed=distributed)
    net, init_step = cmf.factory_model(config, M, dsets["train"], it_logger)
    if distributed:
        net_ddp = torch.nn.parallel.DataParallel(
            net, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )
    
    # Prepare tensorboard
    net.create_tensorboard_summary(config["misc"]["tensorboard_dir"])

    """ Run training network """
    eval_every = config["evaluation"].get("every_eval", 1) # epoch
    eval_after= config["evaluation"].get("after_eval", 0) # epoch
    print_every = config["misc"].get("print_every", 1) # iteration
    num_step = config["optimize"].get("num_step", 30) # epoch
    apply_cl_after = config["model"].get("curriculum_learning_at", -1)

    vis_every = config["misc"].get("vis_every", -1) # epoch
    if vis_every > 0:
        nsamples = config["misc"].get("vis_nsamples", 12)
        vis_data = dsets["train"].get_samples(int(nsamples/2))
        vis_data.extend(dsets["test"].get_samples(int(nsamples/2)))
        vis_data = dsets["train"].collate_fn(vis_data)
        vis_inp, vis_gt = net.prepare_batch(vis_data)
        net.visualize(vis_inp, vis_gt, "epoch{:03d}".format(0))

    # We evaluate initialized model
    #cmf.test(config, L["test"], net, 0, eval_logger, mode="Valid")
    ii = 1
    net.train_mode() # set network as train mode
    net.reset_status() # initialize status
    tm = timer.Timer() # tm: timer
    if local_rank == 0:
        print("=====> # of iteration per one epoch: {}".format(len(L["train"])))
    for epoch in range(init_step, init_step+num_step):
    #for epoch in range(init_step, init_step+3):
        # curriculum learning
        if (apply_cl_after > 0) and (epoch == apply_cl_after):
            net.apply_curriculum_learning()

        for batch in L["train"]:

            # Forward and update the network
            data_load_duration = tm.get_duration()
            tm.reset()
            net_inps, gts = net.prepare_batch(batch)
            if distributed:
                net_out = net_ddp.forward(net_inps)
                loss = net.loss_fn(net_out, gts, count_loss=True)
                net.update(loss)
                outputs = {"loss": loss, "net_output": net_out}
            else:
                outputs = net.forward_update(net_inps, gts)
            
            run_duration = tm.get_duration()

            # Compute status for current batch: loss, evaluation scores, etc
            net.compute_status(outputs["net_output"], gts)

            # print learning status
            if (print_every > 0) and (ii % print_every == 0) and local_rank==0:
                net.print_status()
                lr = net.get_lr()
                txt = "fetching for {:.3f}s, optimizing for {:.3f}s, lr = {:.5f}"
                it_logger.info(txt.format(data_load_duration, run_duration, lr))

            # for debugging
            if config["misc"]["debug"] and (ii > 2):
                cmf.test(config, L["test"], net, 0, eval_logger, mode="Valid")
                cmf.test(config, L["phrase"], net, 0, eval_logger, mode="Phrase")
                break

            tm.reset(); ii = ii + 1
            # iteration done

        # visualize network learning status
        if (vis_every > 0) and (epoch % vis_every == 0):
            net.visualize(vis_inp, vis_gt, "epoch{:03d}".format(epoch))

        # validate current model
        if (epoch > eval_after) and (epoch % eval_every == 0):
            # print training losses
            net.save_results("epoch{:03d}".format(epoch), mode="Train")
            if local_rank == 0:
                net.print_counters_info(eval_logger, epoch, mode="Train")

            if distributed:
                cmf.test(config, L["test"], net_ddp, epoch, eval_logger, mode="Valid", if_print=False, distributed=True)
                cmf.test(config, L["phrase"], net_ddp, epoch, eval_logger, mode="Phrase", if_print=False, distributed=True)
            else:
                cmf.test(config, L["test"], net, epoch, eval_logger, mode="Valid", if_print=False)
                cmf.test(config, L["phrase"], net, epoch, eval_logger, mode="Phrase", if_print=False)
            net.train_mode() # set network as train mode
            net.reset_status() # initialize status



if __name__ == "__main__":
    # get parameters from cmd
    params = _get_argument_params()
    if params["distributed"]:
        torch.cuda.set_device(params["local_rank"])
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        
    global M, dataset
    M, dataset, config = cmf.prepare_experiment(params)

    # train network
    train(config, params["local_rank"], params["distributed"])
