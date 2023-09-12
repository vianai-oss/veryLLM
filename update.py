# (c) 2023 Vianai Systems, Inc.
import argparse

from updates.update_validator import update_validator
from updates.update_leaderboard import (
    auto_update_leaderboard,
    manual_update_leaderboard,
    update_validator_score,
)

parser = argparse.ArgumentParser(
    description="Update the current validator and leaderboard"
)

parser.add_argument(
    "-m",
    "--manual",
    help="Name of manual leaderboard function to update",
)
parser.add_argument(
    "-a",
    "--auto",
    action="store_true",
    help="Update the leaderboard for all new functions",
)
parser.add_argument(
    "-v",
    "--validator",
    action="store_true",
    help="Update the validator",
)
args = vars(parser.parse_args())

if args["manual"]:
    print(f"Updating leaderboard for {args['manual']}\n")
    manual_update_leaderboard(args["manual"])
elif args["auto"]:
    print("Updating leaderboard for all new functions\n")
    auto_update_leaderboard()

if args["validator"]:
    print("Updating validator")
    update_validator()
    update_validator_score()
