# NeuroToy
My playground for machine learning in Unity, in which I figure out how to build something like TensorFlow from scratch.

Experiments contained within:
- Evolving physically simulated creature brains with evolutionary strategies
![Stippy Step](https://i.imgur.com/FGawzUs.gif)
![Slippery Slope](https://i.imgur.com/WmOhsbx.gif)
![Bulky](https://i.imgur.com/M8PQiJq.gif)

- Multi-Layer-Perceptrons and Convolutional Neural Nets built on the new Burst compiler framework and Job System.
![MNIST](https://i.imgur.com/VKNlSg4.png)

Don't forget to turn off Burst Leak Detection and SafetyChecks under the Jobs menu item if you're interested in running this really fast.

To be done:
- Fully automatic differentiation for arbitrary network architectures
- Take the fast Burst code and apply it the older creature experiments
