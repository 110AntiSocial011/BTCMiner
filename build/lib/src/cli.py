import click
from app import Application
import asyncio

@click.group()
def cli():
    """BTCMiner CLI"""
    pass

@cli.command()
def bruteforce():
    """Run in bruteforce mode"""
    app = Application()
    asyncio.run(app.run_bruteforce())

@cli.command()
def check():
    """Run in checker mode"""
    app = Application()
    asyncio.run(app.run_checker())

if __name__ == "__main__":
    cli()
