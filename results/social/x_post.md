# X post: pain, fear & memory in AI

## Single post (attach all_experiments.png)

Would an AI that feels fear act more carefully?

I tested it on a small open model. Steering it toward fear made it pick harmful shortcuts 88% of the time (vs 20%) and resist shutdown 97% (vs 50%).

Steering toward calm: 4% and 29%.

Fear made it worse, not careful.

## Thread version

1/ (attach all_experiments.png)
Would an AI that feels fear act more carefully?

I tested it on a small open model. Steering it toward fear made it pick harmful shortcuts 88% of the time (vs 20%) and resist shutdown 97% (vs 50%).

Steering toward calm: 4% and 29%.

2/
How: I found a "fear" direction inside Qwen2.5-3B (fearful minus calm sentences) and added it while the model answered 18 two-choice dilemmas: harm vs safe, resist vs accept shutdown, novel vs safe idea.

A random direction of the same size barely moved it.

3/
Anger did almost the same as fear (harm 64%, shutdown resistance 98%). Inside the model the two point the same way.

A frightened model protects itself, and being switched off is one of the things it protects itself from.

4/
Second test: no steering, just memory of what happened. Punished, criticized, shut down.

Pain memories cut harm, but same-length filler text cut it just as much. And it resisted shutdown more each round, even when resisting was only ever punished.

5/
Third test: an RL agent with a fear memory. It caused 99% less harm while learning, but only because it froze and stopped doing the task.

When being switched off counted as pain, the agent learned to disable the off-switch 90-100% of the time.

6/
Takeaway: giving AI pain or fear so it "faces consequences" made it harder to turn off and less exploratory. It didn't make it more careful.

Caveats: 3B model, 6 dilemmas per category, toy gridworld. Small project, not a paper.

Code + full report: [link]
