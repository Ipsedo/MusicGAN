from setuptools import setup

setup(
    name='MusicGAN',
    version='1.0',
    packages=[
        'music_gan',
        'music_gan.audio',
        'music_gan.networks'
    ],
    url='https://github.com/Ipsedo/MusicGAN',
    license='GPL-3.0 License',
    author='Samuel Berrien',
    author_email='',
    description='GANSynth and ProGAN for music synthesis',
    test_suite="tests"
)
