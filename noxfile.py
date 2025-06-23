import nox


nox.options.default_venv_backend = "uv"

@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
# @nox.session(python=["3.13"])
def tests(session):
    session.install(".",)
    session.install("pytest", "pytest-cov")
    session.run("pytest")
