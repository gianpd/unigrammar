# MyPerformances Grammar Correction and Questions-Answering Tool

<p>The Grammar correction help both students and professors in their essay.<p>

The tool is composed by a Deep Neural Network model, a transformer ones. Transformer models have been introduced in [1]. Its main feature is about taking into account the similarity between a single word to the rest of a sentence, by applying the Self-Attention Head layer, which returns a weighted version of the sentence. 
<p>
In a nutshull, the self-attention head layer performs the following:
<br>
outputs = sum(inputs * pairwise_scores(inputs, inputs)).
Where, the pairwise_scores is, very often, the dot product between two vectors. 

<p>This means “for each token in inputs (A), compute how much the token is related to
every token in inputs (B), and use these scores to weight a sum of tokens from
inputs (C).” Crucially, there’s nothing that requires A, B, and C to refer to the same
input sequence. In the general case, you could be doing this with three different
sequences. We’ll call them “query,” “keys,” and “values.”<p>

<p>
<b>Why is the dot product suitable as similarity metric?</b>
Let A,B be two vectors, their dot product, can be written as: 
(A,B) = |A||B|cos(Θ), where Θ is the angle between them. If two vectors are ortoghonal, their dot product is zero, meaning they have nothing in common. If the two vectors, A and B, are normalized, in order to be unit vectors, s.t: 
<br>

$$
A' = \frac{A}{|A|} \Rightarrow (A', A') = 1 \Rightarrow (A,B)=cos(Θ)
$$






<br>
[1]: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017)