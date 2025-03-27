# Training a model

## Train loss
> Each token is predicted by selecting an index in our word embedding. In our [trainer script](./trainer.py) we have 50304 unique possible tokens to select from. So if each word were to have equal probability as at start of training all words should be treated the same then our log should be around: -ln(1/50304) ==> 10.82

we start at 11 ___ but quickly below 6 __ which is expected.

## Val loss
> we want val loss to be slightly higher than train loss while both of them are decreasing.
> high train loss + high val loss = not learning
> decreasing train loss + decreasing val loss = learning
> decreasing train loss + constant val loss = overfitting (we almost want to)
> not sure what to write for under fitting.

## target gpt2 / train gpt3
we know openai's gpt2 validation performance by ballpark to be: 2.85
Lets draw a line to which iteration we cross it in. with that we can estimate how many tokens it took us to reach there, helping us evaulate the dataset too.



