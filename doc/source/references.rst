############
 References
############

**********
 DeepMind
**********

The AlphaZero algorithm was introduced by DeepMind in:

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M.,
Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap,
T., Simonyan, K., & Hassabis, D. (2018). "Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning Algorithm." Science,
362(6419), 1140–1144. https://doi.org/10.1126/science.aar6404

***********
 OpenSpiel
***********

The neural network oracles are adapted from the OpenSpiel library:

Lanctot, M., Lockhart, E., Lespiau, J.-B., Zambaldi, V., Upadhyay, S.,
Pérolat, J., Timbers, F., Tuyls, K., Omidshafiei, S., Muller, P.,
Batista, N., Baker, B., Destin, D., et al. (2019). "OpenSpiel: A
Framework for Reinforcement Learning in Games."
https://arxiv.org/abs/1908.09453

Source code: https://github.com/google-deepmind/open_spiel

***********
 TrueSkill
***********

TrueSkill is used to rate saved AlphaZero checkpoints during model
selection. The current best model is chosen by applying a conservative
penalty to the rating uncertainty.

Project repository: https://github.com/sublee/trueskill

********
 KataGo
********

The idea of distributing training across multiple processes that
communicate via files was inspired by KataGo:

Wu, D. J. (2019). "Accelerating Self-Play Learning in Go."
https://arxiv.org/abs/1902.10565

Project repository: https://github.com/lightvector/KataGo

*******
 Codex
*******

This project was developed with the assistance of `OpenAI Codex
<https://openai.com/codex/>`_ from release v0.1.0 through v0.2.0.

Codex highlighted critical bugs that, once resolved, improved training
outcomes.
