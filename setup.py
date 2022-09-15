from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools
from wheel.bdist_wheel import bdist_wheel
import glob
import os
assert parse_version(setuptools.__version__)>=parse_version('36.2')

class CommandBdistWheel(bdist_wheel):
    
    # Called almost exactly before filling `.whl` archive
    def write_wheelfile(self, *args, **kwargs):
        dr = f"{self.bdist_dir}/<package name>"
        paths = [
            path for path in glob.glob(f'{dr}/**/*.py', recursive=True)
            if os.path.basename(path) != '__init__.py'
        ]
        for path in paths:
            os.remove(path)
        super().write_wheelfile(*args, **kwargs)

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected:
    assert o in cfg, f"missing expected setting: {o}"
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 3.10'.split()

requirements = cfg.get('requirements','').split()
lic = licenses[cfg['license']]
min_python = cfg['min_python']


setuptools.setup(
    cmdclass={'bdist_wheel': CommandBdistWheel},
    name=cfg['lib_name'], 
    license=lic[0], 
    classifiers=['Development Status :: ' + statuses[int(cfg['status'])], 
    'Intended Audience :: ' + cfg['audience'].title(), 
    f'License :: {lic[1]}', 
    'Natural Language :: ' + cfg['language'].title()] + [f'Programming Language :: Python :: {o}' for o in py_versions[py_versions.index(min_python) :]], 
    url=cfg['git_url'], 
    packages=setuptools.find_packages(),
    include_package_data=True, 
    install_requires=requirements, 
    dependency_links=cfg.get('dep_links', '').split(), 
    python_requires=f'>={min_python}', 
    long_description=open('README.md', encoding='utf8').read(), 
    long_description_content_type='text/markdown', 
    zip_safe=False, 
    entry_points={'console_scripts': cfg.get('console_scripts', '').split()}, 
    **setup_cfg
)

