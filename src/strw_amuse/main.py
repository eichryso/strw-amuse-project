"""
Example version checking script.
"""
import amuse
import amuse._version

def version_check() -> str:
    '''Display the version of AMUSE.'''
    print(f'AMUSE on v: {amuse._version.__version__}')

if __name__=='__main__':
    version_check()
