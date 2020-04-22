vX_Y: small change w.r.t vX, new file named vX_Y created.

vX.Y: small change within vX, no new file created

0419-173607:
* 30k data, clip to max{reward, -10}
* kl weight: 1e-3

0419-155438:
* same
* kl weight: 1e-2

0416-214954:
* same
* except kl weight: 0.1

0422-154925:
* train-v0_2
* 300k data, gen-v3.1
* training type: freeze
* kl weight: 1e-3
* early stop 20, shuffle 400k
* dgx.slurm 36 nodes

0422-162352:
* same
* except early stop 5, shuffle 20k
* dgx.slurm 24 nodes