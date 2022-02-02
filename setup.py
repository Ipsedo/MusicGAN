from setuptools import setup

setup(
    name='MusicGAN',
    version='1.0',
    packages=['music_gan'],
    url='https://github.com/Ipsedo/MusicGAN',
    license='GPL-3.0 License',
    author='Samuel Berrien',
    author_email='',
    description='GANSynth and ProGAN with bark scale for music synthesis',
    entry_points={
        'console_scripts': ['music_gan = music_gan.__main__:main']
    }
)
