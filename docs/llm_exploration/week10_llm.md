## Session Focus

This week, we moved to tune kernel hyperparameters across a large set of slices. This was a huge problem computationally and the session would often fail. Even with the use of GPUs, the training cell would also take around 20+ minutes to run. My LLM session explored how to make hyperparameter tuning computationally feasible for multiple slices.
Surprising Insights
The cost came from repeatedly solving the full kernel optimization problem for each trial, meaning runtime scaled with both the number of trials and the number of slices
Running full hyperparameter searches across all slices was unnecessary, since many slices share similar structure
Reducing the search space (fewer trials, smaller parameter ranges) led to quicker search

## Techniques that Worked
Reducing the number of trials decreased runtime and still produced reasonable parameter estimates
Tuning on a subset of slices rather than the full dataset also allowed for quicker runtime and a good approximation
Restricting parameter ranges prevented the optimization from exploring regions that did not make a lot of sense in a final reconstruction
Lowering maximum iterations during tuning reduced training time

## Dead Ends
ChatGPT needed a lot of information from the notebook of what was already set up. This led to multiple dead ends because it would give suggestions that did help that much because it didn’t have all information.

## Next Steps
Since we are moving towards general images rather than just MRI images, we will have to apply this setup to general images. The kernel tuning for the MRIs was able to run and we didn’t need to run it multiple times, so Chat’s suggestions did help. Although, if we get to a place where we need to run the tuning a lot, we will have to make further adjustments to make it more computationally feasible.
