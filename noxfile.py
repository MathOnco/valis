from nox_poetry import session

@session(python=["3.9", "3.10", "3.11", "3.12"])

def tests(session):
    session.install('pytest', ".")
    session.run('pytest')