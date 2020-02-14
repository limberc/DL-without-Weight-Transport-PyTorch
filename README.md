# Deep Learning Without Weight Transport

Mohamed Akrout, Collin Wilson, Peter C. Humphreys, Timothy Lillicrap, Douglas Tweed

Current algorithms for deep learning probably cannot run in the brain because they rely on weight transport, where forward-path neurons transmit their synaptic weights to a feedback path, in a way that is likely impossible biologically. In this work, we present two new mechanisms which let the feedback path learn appropriate synaptic weights quickly and accurately even in large networks, without weight transport or complex wiring. One mechanism is a neural circuit called a weight mirror, which learns without sensory input, and so could tune feedback paths in the pauses between physical trials of a task, or even in sleep or in utero. The other mechanism is based on a 1994 algorithm of Kolen and Pollack. Their method worked by transporting weight changes, which is no more biological than transporting the weights themselves, but we have shown that a simple circuit lets forward and feedback synapses compute their changes separately, based on local information, and still evolve as in the Kolen-Pollack algorithm. Tested on the ImageNet visual-recognition task, both the weight mirror and the Kolen-Pollack circuit outperform other recent proposals for biologically feasible learning — feedback alignment and the sign-symmetry method — and nearly match backprop, the standard algorithm of deep learning, which uses weight transport.

## The two new proposed algorithms

Weight Mirrors (WM): it represents the the second learning mode alternating with the engaged mode during the training. This algorithm suggests that neurons can discharge noisily their signals and adjust the feedback weights so they mimic the forward ones. Here is a pseudo-code of this method:

Kolen-Pollack algorithm (KP): it solves the weight transport problem by transporting the changes in weights. At every time step, the forward and backward weights undergo identical adjustments and apply identical weight-decay factors as described in the equations 16 and 17 of the paper manuscript.

Credit

https://github.com/makrout/Deep-Learning-without-Weight-Transport