# Quantifying offensive passing ability with Expected Primary Assists
Dan Morse
3/5/2021

## Introduction

Points are the long-standing measure of talent in hockey. After all scoring goals is better than...not scoring goals. But goals and assists are far from perfect measures.

Recently, the hockey analytics community has instead focused on metrics like **expected** goals, derived primarily from shot location data. Expected goals (xG) have proven to be more predictive of future success than goals alone.

But what about assists?

At times, assists can be the focal point of a goal. [This pass](https://twitter.com/StevenEllisTHN/status/1158374638403149824?s=20) from Jamie Drysdale is a good example of an important assist. Then there are other instances where an assist is counted despite not really being a huge factor in the goal, like [this Connor McDavid goal](https://www.youtube.com/watch?v=gcfIqClMstY) that credited Tyson Barrie with an assist. These two plays are counted the same for both Drysdale and Barrie, despite the fact that Barrie's "contribution" doesn't even make it into the highlight clip.

Thanks to the Stathletes' Big Data Cup, we've got a litany of passing details that I have attempted to compile into a new model to help distinguish between these two plays: Expected Primary Assists (xPA). Using the provided 40-game sample of Erie Otters data, we'll attempt to properly quantify just how impactful passes like the one from former Otter Jamie Drysdale linked above.

## Method

Calculating the likelihood of a pass leading to a goal is essentially a two-part problem:
  
  1. What is the probability that the pass is completed?
  2. What is the probability that the resulting shot goes in the net?

### Pass Completion Probability

For the first part, we'll construct an expected completion model similar to what you'll find in the [NFL analytics community](https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/). I utilized extreme gradient boosting from the R package [xgboost](https://www.rdocumentation.org/packages/xgboost) for this model. The features selected were some basic game information like time remaining, strength state, and score differential, along with various classifications of some of the more common types of passes. The hyperparameters were identified using 5-fold cross validation before being input into the final model.

![pass importance matrix](https://github.com/danmorse314/Expected-Primary-Assists/blob/main/figures/pass%20model%20feature%20importance.png)
*Fig. 1*

The distance traveled by the intended pass is far and away the most important feature in this model. Strength state, which in this case is simply number of skaters for the passing team minus the number of skaters for the opposing team (ie a 5-on-4 powerplay yields a strength of 1). Whether a pass was direct or indirect (indirect being passes off the boards) was also more important than all of our pass classifications. The most significant classification came with our slot pass, which makes sense as it's generally more difficult to find open space right in front of the net to get a pass through/ According to the final model, passes across the slot averaged a 54.2% completion probability vs 73.7% for all other passes.

### Expected Goals

Now that we've got a baseline to determine if a pass will be completed or not, we need to quantify how likely it is that a completed pass will lead directly to a goal. Enter: a new expected goals model.

Expected goals models in the NHL are numerous, but one thing they all omit is passing data. Josh and Luke of [evolving-hockey.com](evolving-hockey.com) opted to include information from events preceding a shot, which captures some of the pre-shot movement, in their model. But the raw NHL data doesn't include passing data. Fortunately for us, the Big Data Cup does.

Alex Novet has [shown in the past](https://hockey-graphs.com/2019/08/12/expected-goals-model-with-pre-shot-movement-part-1-the-model/) that including passing data can indeed improve an expected goals model's accuracy, so we're going to lean into that quite a bit for this one.

![xg importance](https://github.com/danmorse314/Expected-Primary-Assists/blob/main/figures/xg%20model%20feature%20importance.png)
*Fig. 2*

The shooter's proximity to the net remains the most important feature, as is true with all other expected goals models in hockey that I can find. The angle to the net of the shot-taker is about as important as a new feature called "shooter movement." Shooter movement only has a value when a pass was immediately preceding the shot. It is definied as the distance between the shot location and the intended pass destination from the previous event.

Pass distance (for shots with a pass immediately prior) and whether or not the shot was a one-timer also prove to be important factors. As they aren't publicly available in full right now, these aren't normally included in public xGoal models.

The AUC (area under the curve) evaluation of the cross-validation results is better than that of the Evolving Wild xG model (0.828 vs EW's 0.782), which is encouraging in that it appears our new features are aiding the model's accuracy. But it's important to remember that this is only a 40-game sample and could be prone to over-fitting. Testing on future seasons would give us a better idea if this is the case.

We now have the two pieces we originally set out for when searching for xPA: pass completion probability and expected goals. We now have the two pieces we originally set out for when searching for xPA: pass completion probability and expected goals. We can combine to estimate the likelihood of a pass turning into a primary assist with the formula

<img src="https://render.githubusercontent.com/render/math?math=xPA=P(CP)*xG_2">

where <img src="https://render.githubusercontent.com/render/math?math=P(CP)"> is the probability of a completed pass and <img src="https://render.githubusercontent.com/render/math?math=xG_2"> is the expected goals of a shot from the receiver's location.

But what about passes to dangerous areas that are, for whatever reason, incomplete and don't have a shot recorded afterwards? Or passes that are completed but the recipient decides not to shoot?

We can account for these cases by calculating the expected goals value of *theoretical* shots taken after a pass. Our data just needs some small adjustments, and some educated guesses. We can adjust most of the variables fairly easily to account for the new phantom shot, but five others require some extra work:

  *   shot type
  *   traffic
  *   one-timer
  *   shooter movement
  *   time since pass
  
Shot type, traffic, one-timer, and time since pass were all selected at random, weighed by how often they occured in our full dataset of actual shots taken. Shooter movement was sampled from all the shots within one standard deviation of the mean shooter movement in our shots data, and adjusted to ensure all shot locations remained within the bounds of the ice sheet.

The variables were randomly selected and the expected goals calculated 500 times, and with the average xG value being taken for each phantom shot.

##    Results

With that all taken care of, we can now calculate the expected goals and assists of every pass & shot in this data, and compare it to the goals, primary assists, and secondary assists scored on the OHL's [official website](https://ontariohockeyleague.com/stats). I've narrowed it down to 5-on-5 stats as that's when the majority of the game is played, and other situations change the play style dramatically.

![gt table of stats](https://github.com/danmorse314/Expected-Primary-Assists/blob/main/figures/otters%20ev%20stats.png)
*Fig. 3*

It looks like we're measuring something close to primary assists! A linear model shows that xPA explains around 57% of the variance in an individual player's primary assist total at 5-on-5 and around 77% in all situations.

One of the more noticable differences is in how stark a contrast there is between forwards and defensemen. Taking each player's expected primary assists per 100 passes made (xPA/100), the best defenseman on the team is still a less dangerous passer than the worst forward on the team. That makes sense considering defensemen are making a significant portion of their passes on the perimiter of the offensive zone, where the expected goals are low, while forwards send more passes to the middle or down low.

On a contextual level, xPA/100 is measuring how "dangerous" a player's passes were on average.

Among forwards on the Otters last year, Hayden Fowler stands out as someone who might warrant more attention. His xPA/100 was seconds on the team, but he only came away with 4 primary assists. The discrepency makes sense when you see his pass completion percentage over expectation (CPOx) is -8.4%. If we assume that will regress towards the mean next year, we could view Fowler as an under-the-radar player set for a breakout playmaker season. However, there is the chance that CPOx is a skill-based trait, which would mean that while he has an aggressive mindset when it comes to sending passes into dangerous areas, he's simply not very good at actually making the pass. Measuring the season-to-season correlation of this statistic in the future would help give us a better understanding of what to expect in future years.

Contextually, we can also use this as a stylistic measure. Plotting the xPA rate and the xG rate gives us a good idea of which players prefer to shoot from dangerous areas and which prefer to pass.

![fwds style](https://github.com/danmorse314/Expected-Primary-Assists/blob/main/figures/forward%20styles.png)
*Fig. 4*

The Otters' leading goal scorer (both in expected and observed goals) was Chad Yetman. Yetman was also the 2nd least likely forward to pass into dangerous areas on the team. Meanwhile, his frequent linemate Maxim Golod was on the opposite end of the spectrum, opting to pass more often than all but one other forward on the team. The pair playing on the same line produced the most goals on the team last year, with Golod netting the primary assist on nearly half of Yetman's goals at even strength.

![yetman alluvial](https://github.com/danmorse314/Expected-Primary-Assists/blob/main/figures/yetman%20alluvial.png)
*Fig. 5*

Coaches frequently attempt to make forward lines with good chemistry. A breakdown like one in Fig. 4 could aid in that search, as we've already seen that pairing a player whose game is favored by xPA with a player who looks great by xG can lead to a winning combination. Based on this data, the play styles of Hayden Fowler and Elias Cohen should mesh very well together. Giving them more playing time together should lead to more high-danger shots by Cohen on high-danger passes from Fowler.

In the future, the passing model could use more fine tuning. Perhaps adding a new feature like whether or not the pass was sent through traffic would give it a boost. As of now, the xPA model can already be helpful in evaluating different player styles. With year-over-year testing, it will also benefit player evaluation once we determine which elements are skill-based and which are likely to regress towards the mean.
