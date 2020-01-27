# Thesis
### The Racial Voting Power Gap: Analyzing Racial Gerrymandering Through Solving Ecological Inference with The Discrete Voter Model

Hakeem Angulu's undergraduate thesis for the departments of Computer Science and Statistics at Harvard College.

Section 2 of the Voting Rights Act of 1965 (VRA) prohibits voting practices or procedures that discriminate based on race, color, or membership in a language minority group, and is often cited by plaintiffs seeking to challenge racially-gerrymandered districts in court.

In 1986, with Thornburg v. Gingles, the Supreme Court held that in order for a plaintiff to prevail on a section 2 claim, they must show that:

1. the racial or language minority group is sufficiently numerous and compact to form a majority in a single-member district
2. that group is politically cohesive
3. and the majority votes sufficiently as a bloc to enable it to defeat the minority’s preferred candidate

All three conditions are notoriously hard to show, given the lack of data on how people vote by race.

In the 1990s and early 2000s, Professor Gary King’s ecological inference method tackled the second condition: racially polarized voting, or racial political cohesion. His technique became the standard technique for analyzing racial polarization in elections by inferring individual behavior from group-level data. However, for more than 2 racial groups or candidates, that method hits computational bottlenecks.

A new method of solving the ecological inference problem, using a mixture of contemporary statistical computing techniques, is demonstrated here. It is called the **Discrete Voter Model**. It can be used for multiple racial groups and candidates, and is shown to work well on randomly-generated mock election data.

To completely reproduce test results on generated election data, run `methods.ipynb` in full. The requirements can be installed with the included `Pipfile`. Go [here](https://realpython.com/pipenv-guide/) or [here](https://pipenv.kennethreitz.org/en/latest/) for more information on how to use `Pipenv`.
