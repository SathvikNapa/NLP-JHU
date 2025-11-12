### Questions

Q2

(a) Why does Algorithm 1 initialize αbos(0) and βeos(n) to 1?

Answer:

* We set alphaBOS(0)=1 since we need to start all valid paths at BOS.
* We set betaEOS(n)=1 since all valid paths need to end at EOS.

As mentioned in the section "Don’t guess when you already know", we do this specifically to not meddle with the
conditional probabilities of actual transitions and emissions. We ensure
that we do not add the EOS and BOS to emission matrix B. We add the BOS and EOS to A since we need have all valid
transitions. The values for the BOS will be -inf in log-space and 0 in non-log probability space. EOS is unused and
hence we make it np.nan

(b) If you train on a sup file and then evaluate on a held-out raw file, you’ll get lower perplexity
than if you evaluate on a held-out dev file. Why is that? Which perplexity do you think is
more important and why?

* The perplexity of the hmm model trained on ensup data produces lower perplexity on enraw data than the endev data. The
  number of records observed in the endev dataset is too few to make a fair comparison against the enraw unlabeled data,
  which has 4x number of records.
* The perplexity of the model when evaluated on ensup4k is ~534 much lesser than the other two datasets, which might be
  due to the presence of similar tagged sentences in the evaluation dataset to what the model was trained on.
* The perplexity of the hmm model trained on czsup data produces lower perpexity on the dev data than the czraw data.

The reason for lower perplexity on raw data than the labeled dev data:
> During the evaluation of model on raw data, the score computation block marginalizes over all possible tags and hence
> it might result in lower perplexity.
> During the evaluation of model on dev data, the score computation block scores on the joint distribution of word and
> tags, which might be higher.

Which perplexity do you think is more important and why?
> For language modeling/ language generation, the perplexity on the raw data matters more.
> For the supervised tagging task, the perplexity on the dev data matters more. But ultimately, the accuracy and F1
> score give us a better picture than the perplexity on the dev data.

EnSup

```commandline
Eval on Labeled Dev:
INFO:eval:Cross-entropy: 9.9119 nats (= perplexity 20169.835)
INFO:eval:Tagging accuracy: all: 90.455%, known: 96.786%, seen: nan%, novel: 24.858%

Eval on ensup4k:
INFO:eval:Cross-entropy: 6.2818 nats (= perplexity 534.756)
INFO:eval:Tagging accuracy: all: 97.771%, known: 97.771%, seen: nan%, novel: nan%

Eval on Unlabeled Raw:
63.31s user 10.99s system 166% cpu 44.548 total
INFO:eval:Cross-entropy: 9.7547 nats (= perplexity 17235.596)
```

Model trained on EnSup25k (More or less the same performance on the dev data. Better perplexity on raw data)

```commandline
Eval on Dev
INFO:eval:Cross-entropy: 13.2078 nats (= perplexity 544605.062)
INFO:eval:Tagging accuracy: all: 84.158%, known: 96.327%, seen: nan%, novel: 23.869%

Eval on Raw
INFO:eval:Cross-entropy: 12.6742 nats (= perplexity 319411.085)
```

ICSup model

```commandline
icsup trained model on icdev data
> INFO:eval:Cross-entropy: 1.2864 nats (= perplexity 3.620)

icsup trained model on icraw data
> INFO:eval:Cross-entropy: 1.2217 nats (= perplexity 3.393)
```

CZSup model

```commandline
czsup trained model | eval on czdev data
> INFO:eval:Cross-entropy: 15.2758 nats (= perplexity 4307265.420)

czsup trained model | eval on czdev data
> INFO:eval:Cross-entropy: 13.6666 nats (= perplexity 861638.465)
```

(c) V includes the word types from sup and raw (plus oov). Why not from dev as well?

```commandline
In order to not let the model cheat by looking at the data we evaluate on and not help the model unfairly.
The dev file should test on OOV handling of unseen text/words and generalize to the data unobserved.
The dev set is a simulated real-world data that helps us evaluate the model trained on the train data. 
```

(d) Did the iterations of semi-supervised training help or hurt overall tagging accuracy? How
about tagging accuracy on known, seen, and novel words (respectively)?

```commandline
It hurt the overall tagging accuracy (90.455% to 79.987%)
Tagging on known has made a significant drop from 96.78% to 84.309%.

python3 tag.py ../data/endev --model ../models/enunsup.pkl --checkpoint        54.24s user 6.48s system 154% cpu 39.184 total

INFO:eval:Cross-entropy: 12.3305 nats (= perplexity 226496.468)
INFO:eval:Tagging accuracy: all: 79.987%, known: 84.309%, seen: nan%, novel: 35.199%
```

```commandline
Supervised Model:
INFO:eval:Tagging accuracy: all: 90.455%, known: 96.786%, seen: nan%, novel: 24.858%

Semisupervised Model:
INFO:eval:Tagging accuracy: all: 93.595%, known: 96.566%, seen: nan%, novel: 62.808%
```

* The model trained on semi-supervised training helped improve the overall tagging accuracy from 90.45% to 93.59%.
* While there is a very slight drop of accuracy on known data (96.786% to 96.566%), the accuracy on the novel data
  jumped significantly from (24.8% to 62.8%).

(e) Explain in a few clear sentences why the semi-supervised approach might sometimes help.
How does it get additional value out of the enraw file?

Answer:

### Empirical Analysis

* The semisupervised HMM model achieved a very minimal accuracy gain (+3.27%) by refining emission probabilities for
  unknown words through fractional counts from enraw.
* About 80% of changed predictions were improvements, showing that unlabeled data mainly
  helped the model generalize a bit to the unseen tokens rather than overfitting to the supervised corpus.
* enraw (Model the semisupervised model trained/adapted on) provided extra contexts for rare and OOV words, allowing EM
  to assign relative frequency estimates to plausible tags based on distributional/neighboring evidence.

#### Analysis

* Consider Supervised Model as ModelA and Semisupervised model as ModelB.
* ModelA was evaluated on 23949 tokens. ModelB helped fix around **976** tagging errors while making **224** tagging
  errors (regression). Total number of unchanged token tagging was about 22749 tokens.

* Model 1 has an Accuracy of 95.93%, with 22,973 True positives, 976 FN and FP.
* Model 2 has an accuracy of
  Accuracy = 99.065% ⇒ TP = 23,725, FP = FN = 224

#### Shifts in probabilities

* The emission probability matrix B observed more shifts in the parameters than the transition matrix A.
* The shift in probabilities is observed by max element-wise change of probabilities (not in log space)

> max(Change in A) = 0.175 (17.5 percentage points)
> max(Change in B) = 0.658  (65.8 percentage points).

This shift in probabilities shows us that the semisupervised model changed the emission probabilities more than the
transition properties, as mentioned in the handout. The lesser change in transition probability might be because the
supervised model might have already learned the transitions well from the labels.

#### Most fixed token

* Nearly all or most correct corrected tokens were _OOV_ tokens. (Meaning, the unlabeled dataset on which the
  semisupervised model might have had new context than what the ensup data had)
* This conversion of had the biggest contribution to the emission probability change.

(f) Suggest at least two reasons to explain why the semi-supervised approach didn’t always help.

While semisupervised learning helps learn the unobserved patterns in the prior supervised learning, they can degrade the
performance sometimes because of the statistically mismatched or noisy data from train and other challenges:

* The EM algorithm, as we observed with the ice-cream example in class, is not always guaranteed to find the global
  maxima/minima. The loss function is not convex and hence it can get stuck at a local optimum.
* The EM algorithm is inertial in nature, meaning the probability of the current state is influenced by the previous
  states. If the initial parameters are random and has a bias to a specific tag for a word, the E-step on raw data will
  reinforce this bias and M-step will set the bias in the params, leading to a suboptimal learning.
* Weak regularization in the optimization objective can also lead to a model biased to the dataset with higher number of
  records. If the unlabeled dataset is large in number and has different statistics as opposed to the labeled base
  model's train data, we will alter the belief of the model and will model it close to unlabeled data, which miht be
  problematic for tagging task. The model biased by unlabeled data might understand the underlying relationships of the
  text observed but might not be well-equipped to handle a specific tagging task as it might not know that Papa needs to
  be PN, but might know that Papa appears closely with "eats" and eats with "caviar".

g. How does your bigram HMM tagger compare to a baseline unigram HMM tagger (see reading
section G.3)? Consider both accuracy and cross-entropy. Does it matter whether you use
enraw?

```commandline
Supervised Unigram Model trained on ensup evaluated on endev
INFO:eval:Tagging accuracy: all: 87.674%, known: 95.980%, seen: nan%, novel: 1.613%
INFO:eval:Cross-entropy: 13.8220 nats (= perplexity 1006515.912)

Semisupervised Unigram Model trained on ensup and enraw. Evaluated on endev 
INFO:eval:Tagging accuracy: all: 87.540%, known: 95.980%, seen: nan%, novel: 0.095%
INFO:eval:Cross-entropy: 13.8220 nats (= perplexity 1006516.458)

Bigram Model trained on ensup evaluated on endev
INFO:eval:Tagging accuracy: all: 90.455%, known: 96.786%, seen: nan%, novel: 24.858%
INFO:eval:Cross-entropy: 9.9119 nats (= perplexity 20169.835)
```

The bigram model shows a sharp improvement to our baseline unigram model in terms of tagging accuracy and language
generation.

Tagging:

* The unigram model observes a slight drop (overall: 87.67% and known: 96%) as opposed to Bigram's (overall: 90.5% and
  known: 96.78%), the accuracy on the novel data shows a significant drop from
  an already lower 24% to 1.6%.
* The semi-supervised training of the unigram only worsens the tagging accuracy of tagging on the novel dataset.
* This might be due to the bigram model's markovian assumption that helps it learn local properties that the unigram
  model cannot.
* For novel context, the bigram model seems to perform better due the dependence/conditioning on the previous tag, which
  the unigram does not observe.

Language Generation:

* The cross-entropy and perplexity numbers of the unigram model are much higher as opposed to the bigram model. The high
  cross-entropy and perplexity numbers shows uncertainty/incoherence in the text generation while the bigram seems to be
  an improvement on the unigram model.

h. Experiment with different training strategies for using enraw. For example, you could train on enraw alone, or a
combined corpus of ensup+enraw, or ensup+ensup+ensup+enraw (so that the supervised data is weighted more heavily). And
you could do staged training where you do additional training on ensup before or after this.
What seems to work? Why?

Answer:

| **Setup**                                    | **Cross-Entropy (nats)** | **Perplexity** | **Accuracy (All)** |
|:---------------------------------------------|:------------------------:|:--------------:|:------------------:|
| **baseline: (ensup only)**                   |          7.2796          |     1450.5     |    **91.07 %**     |
| **weighted (ensup + ensup + ensup + enraw)** |          7.2063          |     1347.9     |      89.72 %       |
| **staged (ensup + enraw + ensup)**           |          7.2129          |     1356.8     |      89.79 %       |
| **Tau-sharpened raw (Tau = 0.7)**            |          7.2062          |     1347.7     |      89.35 %       |
| **enraw-only + tiny ensup**                  |          7.2603          |     1422.6     |      90.23 %       |

* Incorporating enraw provides only marginal benefits without careful weighting or staged fine-tuning. Unlabeled data
  smooths likelihoods but weakens tag precision.
* The baseline with ensup only achieved the gold standard accuracy (~91%). Fully supervised training yields the cleanest
  tag and emission estimates.
* Adding enraw lowered cross-entropy reduced the accuracy but learned smoother distributions from the unlabeled corpus.
* The model weighted on ensup (ensup + ensup + ensup + enraw) and (ensup + enraw + ensup) staged model creation
  processes produced similar accuracy metrics. Showed minor perplexity improvements with 1-1.5% accuracy loss.
* Tau-sharpening has not improved the performance.
* The model training only with enraw with minimal labeled data based supervised training, reached comparable cross
  entropy but has reduced the worse accuracy. This shows that EM can reasonably improve the emission patterns but still
  needs labeled anchors for predicting tags accurately.

4.a) Compare the CRF to the HMM, training on ensup and evaluating on endev. Run a few
experiments to compare them under different conditions (hyperparameters, unigram mode,
features from etc.) and using different metrics (cross-entropy, accuracy). Look at the output
if possible. What did you learn?

Answer:

| **Model** | **Hyperparameters**               | **Cross-Entropy** | **Accuracy** | **Known Acc** | **Novel Acc** |
|:----------|:----------------------------------|:-----------------:|:------------:|:-------------:|:-------------:|
| HMM       | λ = 0 (no smoothing)              |      9.9119       |   90.46 %    |    96.79 %    |    24.86 %    |
| HMM       | λ = 0.5 (mild smoothing)          |      7.1537       |   89.87 %    |    94.54 %    |    41.41 %    |
| HMM       | λ = 1 (moderate smoothing)        |      7.3767       |   88.42 %    |    92.87 %    |    42.32 %    |
| HMM       | λ = 2 (heavy smoothing)           |      7.6768       |   86.61 %    |    90.73 %    |    43.88 %    |
| **CRF**   | LR = 0.05, batch = 30             |    **0.2564**     | **91.03 %**  |    93.68 %    |  **63.57 %**  |
| CRF       | LR = 0.05, batch = 32             |      0.2717       |   90.63 %    |    93.11 %    |    64.90 %    |
| CRF       | LR = 0.01, batch = 32             |      0.3600       |   87.48 %    |    89.67 %    |    64.85 %    |
| CRF       | LR = 0.1, batch = 32              |      0.5340       |   82.00 %    |    83.75 %    |    63.95 %    |
| CRF       | LR = 0.05, batch = 128            |      4.4629       |   61.62 %    |    62.13 %    |    56.40 %    |
| CRF       | LR = 0.5, batch = 256             |      7.6683       |   63.32 %    |    66.47 %    |    30.65 %    |
| CRF       | LR = 0.01, batch = 32, reg = 1e-4 |      0.2716       |   90.64 %    |    93.13 %    |    64.90 %    |

* While training on ensup and evaluating on endev, CRF under various hyperparameters worked better with a significantly
  lower cross entropy (~0.26) and higher overall accuracy (~91%) than any HMM variant observed.
* Smoothing the HMM helped unseen-word accuracy but observed a mild drop on the known data accuracy. However, since we
  prioritize on the accuracy of tagging on unseen data, smoothing seems to be important.
* CRF performance was stable around LR = 0.05 and small batch sizes but with larger batch sizes and learning rates, we
  observed a marked degradation on both metrics.
* CRF's discriminative objective makes it much more effective to the HMM for the tagging task.

4b. What happens when you include enraw in the training data for your CRF? Explain.

CRF Semisupervised training and evaluation on endev

* Cross-entropy: 0.2724 nats (= perplexity 1.313)
* Tagging accuracy: all: 90.480%, known: 92.133%, novel: 65.984%

CRF supervised training and evaluation on endev

* Cross-entropy: 0.2564 nats (= perplexity 1.292)
* Tagging accuracy: all: 91.03%, known: 93.68%, novel: 63.57%

* Adding enraw to the semisupervised CRF objective, slightly reduced overall and known accuracy but marginally increased
  the novel/OOV accuracy, while lowering cross-entropy (0.272). This signals that enraw predominantly refined emission
  features, tending the model to generalize to unseen words at the cost of a small dilution of weights of previously
  observed tokens.

* The tradeoff is an expected behavior of semisupervised objectives. While the objective improves under-represented
  regions, it can introduce mild bias or noise for well-covered tokens unless confidence thresholds and ensup based
  modeling is weight tuned/regularized.

