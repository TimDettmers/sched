from beaker import Beaker

beaker = Beaker.from_env(default_workspace="ai2/timd")

print(beaker.experiment.list(author='timd', finalized=True))


