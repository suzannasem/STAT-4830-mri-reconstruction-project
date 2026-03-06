# LLM Exploratory Summary 1

## Session Focus
Main focus was implementing warm-start neural network using the Gaussian kernel reconstruction

## Surprising Insights
- Deciding what metrics to use and whether to switch NN architecture to U-Net. One pitfall of our project has been that we don’t have medical backgrounds or understandings of what is considered a ‘good’ reconstruction. By using industry-standard architectures and success metrics, we are incorporating the insight of the medical community and getting closer to better reconstruction.
- I had to be very specific about the data that I was working with, what metrics I wanted measured, and how I wanted the neural network to work. For example, Gemini included a conversion from NumPy array to PyTorch tensor in the original function call, but some data was already stored a NumPy array while other images were still in tensor form. I learned that the more specific task, the better.

## Techniques that Worked
- We used a ResidualCorrector function that directly told the neural network not to re-write measured k-space values. This ensures no hallucinations.
- Reduced learning/runtime: using a warm start means the NN has a lot less information to learn and guess. We’re also using the single reconstructed image rather than the full scan stack, which cuts down runtime.
- Troubleshooting implementation errors by just including the Traceback and the error message was not an effective method. Only after providing context (like data types, what was previously defined in the notebook, etc.) was Gemini able to actually troubleshoot these errors.

## Dead Ends
Open-ended phrases (ex. “Explain this to me,” “Help me implement [x]”) didn’t always give me the response I was looking for. For example: “{data} is already in tensor form. Do we still need to convert {data} to a tensor? It’s running an error when I run it.” The response was about how PyTorch only operates on tensors and how we always have to convert to a tensor, which is true, but irrelevant to what I was actually wanting to know (Should we keep the line in the function that converts to a tensor?) Specificity is essential.

# LLM Exploratory Summary 2
## Session Focus
How we can generalize our work to images or videos in general

## Surprising Insights
- MRI images are smoother and more structured than regular images, so a simple Gaussian kernel may oversmooth. This changes our approach to what kernel is best and the one that works best for MRI images may not for regular images.
- We interpret our reconstruction data as undersampled k-space, but that doesn’t translate over to regular images

## Techniques that Worked
I needed to provide the LLM with a decent amount of the work that we had done and on our approach for it to come up with helpful responses

## Dead Ends
Not really any dead ends but that’s mostly because this session was pretty general and I didn’t need to discuss anything super specific.

## Next Steps
Finding a good dataset for videos and exploring how this could be applied there as well.
