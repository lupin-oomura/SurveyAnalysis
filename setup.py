from setuptools import setup, find_packages

setup(
    name='SurveyAnalysis',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'mecab-python3',
        'wordcloud',
        'matplotlib',
        'openai',
        'python-dotenv',
        'scikit-learn',
        'jaconv',
    ],
    url='https://github.com/lupin-oomura/SurveyAnalysis.git',
    author='Shin Oomura',
    author_email='shin.oomura@gmail.com',
    description='Survey of Free Answer Analyzer',
)
