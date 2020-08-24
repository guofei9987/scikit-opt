# Contributing guidelines

This page explains how you can contribute to the development of
scikit-opt by submitting patches, tests, new models, or examples.

scikit-opt is developed on
[Github](https://github.com/guofei9987/scikit-opt) using the
[Git](https://git-scm.com/) version control system.

## Submitting a Bug Report

-   Include a short, self-contained code snippet that reproduces the
    problem
-   Ensure that the bug still exists on latest version.

## Making Changes to the Code

For a pull request to be accepted, you must meet the below requirements.
This greatly helps in keeping the job of maintaining and releasing the
software a shared effort.

-   **One branch. One feature.** Branches are cheap and github makes it
    easy to merge and delete branches with a few clicks. Avoid the
    temptation to lump in a bunch of unrelated changes when working on a
    feature, if possible. This helps us keep track of what has changed
    when preparing a release.
-   Commit messages should be clear and concise. If your commit references or
    closes a specific issue, you can close it by mentioning it in the
    [commit
    message](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue).
    (*For maintainers*: These suggestions go for Merge commit comments
    too. These are partially the record for release notes.)
-   Each function, class, method, and attribute needs to be documented.
-   If you are adding new functionality, you need to add it to the
    documentation by editing (or creating) the appropriate file in
    `docs/`.


## How to Submit a Pull Request

So you want to submit a patch to scikit-opt but are not too familiar
with github? Here are the steps you need to take.

1.  [Fork](https://help.github.com/articles/fork-a-repo) the
    [scikit-opt repository](https://github.com/guofei9987/scikit-opt)
    on Github.
2.  [Create a new feature
    branch](https://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging).
    Each branch must be self-contained, with a single new feature or
    bugfix.
3.  Make sure the test suite passes. This includes testing on Python 3.
    The easiest way to do this is to either enable
    [Travis-CI](https://travis-ci.org/) on your fork, or to make a pull
    request and check there.
4.  If it is a big, new feature please submit an example.
5.  [Submit a pull
    request](https://help.github.com/articles/using-pull-requests)

## License

scikit-opt is released under the MIT license.
