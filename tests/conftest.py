import sys
import platform
print(platform.python_version())


from valis import slide_io
def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    slide_io.init_jvm()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    slide_io.kill_jvm()
