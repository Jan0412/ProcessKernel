"""The ORM: a listwise (LambdaRank) reranker over finished kernels.

The shared base (``encoding``, ``model``, ``dataset``, ``metrics``, ``trainer``) plus the
listwise parts: ``lists`` builds one speed-graded list per problem, ``list_dataset`` and
``list_trainer`` feed and train on them, ``train`` is the entry point.
"""
