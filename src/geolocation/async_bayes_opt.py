import asyncio
import json
import random
import threading
import time

import click
import requests
import tornado.httpserver
import tornado.ioloop
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours, UtilityFunction
from tornado.web import RequestHandler

from geolocation.config import AVAILABLE_LOCI

#
#
# class BayesianOptimizationHandler(RequestHandler):
#     """Basic functionality for NLP handlers."""
#     _bo = BayesianOptimization(
#         f=black_box_function,
#         pbounds={"x": (-4, 4), "y": (-3, 3)}
#     )
#     _uf = UtilityFunction(kind="ucb", kappa=3, xi=1)
#
#     def post(self):
#         """Deal with incoming requests."""
#         body = tornado.escape.json_decode(self.request.body)
#
#         try:
#             self._bo.register(
#                 params=body["params"],
#                 target=body["target"],
#             )
#             print("BO has registered: {} points.".format(
#                 len(self._bo.space)), end="\n\n")
#         except KeyError:
#             pass
#         finally:
#             suggested_params = self._bo.suggest(self._uf)
#
#         self.write(json.dumps(suggested_params))
#
#
# def run_optimization_app():
#     asyncio.set_event_loop(asyncio.new_event_loop())
#     handlers = [
#         (r"/bayesian_optimization", BayesianOptimizationHandler),
#     ]
#     server = tornado.httpserver.HTTPServer(
#         tornado.web.Application(handlers)
#     )
#     server.listen(9009)
#     tornado.ioloop.IOLoop.instance().start()
#
#
# def run_optimizer():
#     global optimizers_config
#     config = optimizers_config.pop()
#     name = config["name"]
#     colour = config["colour"]
#
#     register_data = {}
#     max_target = None
#     for _ in range(10):
#         status = name + " wants to register: {}.\n".format(register_data)
#
#         resp = requests.post(
#             url="http://localhost:9009/bayesian_optimization",
#             json=register_data,
#         ).json()
#         target = black_box_function(**resp)
#
#         register_data = {
#             "params": resp,
#             "target": target,
#         }
#
#         if max_target is None or target > max_target:
#             max_target = target
#
#         status += name + " got {} as target.\n".format(target)
#         status += name + " will to register next: {}.\n".format(register_data)
#         print(colour(status), end="\n")
#
#     global results
#     results.append((name, max_target))
#     print(colour(name + " is done!"), end="\n\n")


@click.command()
@click.option('--radius',
              default=1., help='Target distance')
@click.option('--k',
              default=30, help='Ranked accuracy parameter')
@click.option('--train_path',
              default='default', help='Path to train data')
@click.option('--valid_path',
              default='default', help='Path to validation data')
@click.option('--loci',
              default='default',
              help=f'Expected format: LOCUS1;LOCUS2;...;LOCUSN\n'
                   f'Default loci: {AVAILABLE_LOCI}')
def main(radius, k, train, val, loci):
    """Simple program that greets NAME for a total of COUNT times."""
    for i in range(10):
        click.echo('Hello %s!' % radius)


if __name__ == "__main__":
    main()

    # ioloop = tornado.ioloop.IOLoop.instance()
    # optimizers_config = [
    #     {"name": "optimizer 1", "colour": Colours.red},
    #     {"name": "optimizer 2", "colour": Colours.green},
    #     {"name": "optimizer 3", "colour": Colours.blue},
    # ]
    #
    # app_thread = threading.Thread(target=run_optimization_app)
    # app_thread.daemon = True
    # app_thread.start()
    #
    # targets = (
    #     run_optimizer,
    #     run_optimizer,
    #     run_optimizer
    # )
    # optimizer_threads = []
    # for target in targets:
    #     optimizer_threads.append(threading.Thread(target=target))
    #     optimizer_threads[-1].daemon = True
    #     optimizer_threads[-1].start()
    #
    # results = []
    # for optimizer_thread in optimizer_threads:
    #     optimizer_thread.join()
    #
    # for result in results:
    #     print(result[0], "found a maximum value of: {}".format(result[1]))
    #
    # ioloop.stop()