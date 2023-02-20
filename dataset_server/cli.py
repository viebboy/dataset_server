"""
cli.py: cli to launch dataset server
------------------------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran
* Emails: viebboy@gmail.com
* Date: 2022-06-14
* Version: 0.0.1

License
-------
Apache License 2.0


"""

import argparse
from datetime import date
import os
import asyncio

from dataset_server.server import DataloaderServer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-file",
        required=True,
        type=str,
        help="path to dataset file"
    )
    parser.add_argument(
        "--dataset-parameter-file",
        required=True,
        type=str,
        help="path to parameter file"
    )
    parser.add_argument(
        "--port",
        default=50000,
        type=int,
        help="port for serving. Default to 50000"
    )
    parser.add_argument(
        "--nb-server",
        default=1,
        type=int,
        help="total number of servers. Default to 1"
    )

    parser.add_argument("--server-index", default=0, type=int, help="server index. Default to 0")
    parser.add_argument("--batch-size", default=64, type=int, help="total batch size. Default to 64")

    parser.add_argument(
        "--max-queue-size",
        default=128,
        type=int,
        help="maximum number of samples to preload. Default to 128"
    )
    parser.add_argument(
        "--shuffle",
        default='False',
        choices=['False', 'false', 'True', 'true'],
        type=str,
        help="whether to shuffle. Default to False"
    )
    parser.add_argument(
        "--packet-size",
        default=125000,
        type=int,
        help="size of packet in bytes. Default to 125000"
    )
    parser.add_argument(
        "--status-file",
        required=True,
        type=str,
        help="status file to write the status"
    )

    args = parser.parse_args()
    if args.shuffle in ['True', 'true']:
        args.shuffle = True
    else:
        args.shuffle = False

    return args


def main():
    args = parse_args()
    server = DataloaderServer(
        dataset_module_file=args.dataset_file,
        dataset_params_file=args.dataset_parameter_file,
        total_server=args.nb_server,
        server_index=args.server_index,
        batch_size=args.batch_size,
        max_queue_size=args.max_queue_size,
        port=args.port,
        name=f'DataloaderServer-{args.server_index}',
        shuffle=args.shuffle,
        packet_size=args.packet_size,
        status_file=args.status_file,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.run())

if (__name__ == "__main__"):
    main()
