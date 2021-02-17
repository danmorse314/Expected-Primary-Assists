---
title: "Quantifying offensive passing ability with Expected Primary Assists"
author: "Dan Morse"
date: "3/5/2021"
output:
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(xgboost, exclude = "slice")
library(gt)
library(ggalluvial)
```

## Introduction

Points are the long-standing measure of talent in hockey. After all scoring goals is better than...not scoring goals. But goals and assists are far from perfect measures.

Recently, the hockey analytics community has instead focused on metrics like **expected** goals, derived primarily from shot location data. Expected goals (xG) have proven to be more predictive of future success than goals alone.

But what about assists?

At times, assists can be the focal point of a goal. [This pass](https://twitter.com/StevenEllisTHN/status/1158374638403149824?s=20)^[https://twitter.com/StevenEllisTHN/status/1158374638403149824?s=20] from Jamie Drysdale is a good example of an important assist. Then there are other instances where an assist is counted despite not really being a huge factor in the goal, like [this Connor McDavid goal](https://www.youtube.com/watch?v=gcfIqClMstY)^[https://www.youtube.com/watch?v=gcfIqClMstY] that credited Tyson Barrie with an assist. These two plays are counted the same for both Drysdale and Barrie, despite the fact that Barrie's "contribution" doesn't even make it into the highlight clip.

Thanks to the Stathletes' Big Data Cup, we've got a litany of passing details that I have attempted to compile into a new model to help distinguish between these two plays: Expected Primary Assists (xPA). Using the provided 40-game sample of Erie Otters data, we'll attempt to properly quantify just how impactful passes like the one from former Otter Jamie Drysdale linked above.

## Method

Calculating the likelihood of a pass leading to a goal is essentially a two-part problem:
  
  1. What is the probability that the pass is completed?
  2. What is the probability that the resulting shot goes in the net?

### Pass Completion Probability

For the first part, we'll construct an expected completion model similar to what you'll find in the [NFL analytics community](https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/)^[https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/]. I utilized extreme gradient boosting (xgboost) for this model. The features selected were some basic game information like time remaining, strength state, and score differential, along with various classifications of some of the more common types of passes. The hyperparameters were identified using 5-fold cross validation before being input into the final model.

```{r preparing passing data, include = FALSE}
# read scouting data
bdc_data <- read_csv("https://raw.githubusercontent.com/bigdatacup/Big-Data-Cup-2021/main/hackathon_scouting.csv") %>%
  janitor::clean_names() %>%
  mutate(event_index = row_number())

# get only the passing plays
# convert to all numerical columns
passers <- bdc_data %>%
  filter(event == "Play" | event == "Incomplete Play") %>%
  separate(clock, into = c("minutes","seconds","milliseconds"), sep = ":", remove = FALSE) %>%
  mutate(
    minutes = as.numeric(minutes), seconds = as.numeric(seconds),
    period_seconds_remaining = (60 * minutes) + seconds,
    game_seconds_remaining = period_seconds_remaining + ((period - 1) * (20*60)),
    complete_pass = ifelse(event == "Play", 1, 0),
    home = ifelse(team == home_team, 1, 0),
    team_skaters = ifelse(home == 1, home_team_skaters, away_team_skaters),
    opponent_skaters = ifelse(home == 1, away_team_skaters, home_team_skaters),
    strength = team_skaters - opponent_skaters,
    score_differential = ifelse(home == 1, home_team_goals - away_team_goals, away_team_goals - home_team_goals),
    direct = ifelse(detail_1 == "Direct", 1, 0),
    distance = abs(sqrt((y_coordinate_2 - y_coordinate)^2 + (x_coordinate_2 - x_coordinate)^2)),
    breakout = ifelse(
      # passes out of the defensive zone
      x_coordinate <= 75 & x_coordinate_2 > 75, 1, 0
    ),
    dzone_behind_net = ifelse(
      # regrouping passes behind your own net
      x_coordinate <= 75 & x_coordinate_2 < 11, 1, 0
    ),
    dzone = ifelse(
      # passes contained in the defensive zone, not behind the net
      x_coordinate <= 75 & x_coordinate_2 <= 75 & dzone_behind_net == 0, 1, 0
    ),
    ozone_d2d = ifelse(
      # passes in the offensive zone, above the circles
      between(x_coordinate, 125, 154) & between(x_coordinate_2, 125, 154), 1, 0
    ),
    ozone_behind_net = ifelse(
      #passes that come from behind the net
      x_coordinate > 189 & x_coordinate_2 >= 125, 1, 0
    ),
    slot = ifelse(
        # passes across the 'royal road' below the top of the circles
        # 46 ft to top of the circles in the offensive zone
        x_coordinate > 154 & x_coordinate_2 > 154 &
        # only count passes that cross from one side of the net to the other side
        ((y_coordinate < 42.5 & y_coordinate_2 > 42.5) |
           (y_coordinate > 42.5 & y_coordinate_2 < 42.5)),
      1, 0
    ),
    neutral_zone = ifelse(
      # passes entirely in the neutral zone
      between(x_coordinate, 76, 124) & between(x_coordinate_2, 76, 124), 1, 0
    )
  )

# get index for passes for joins to full dataset later
pass_index <- select(passers, event_index)

# keep only my features
pass_data <- passers %>%
  select(
    strength, score_differential:distance, breakout:neutral_zone, complete_pass
  )

pass_model_feats <- names(pass_data)
```

```{r training pass model, echo = FALSE, fig.cap = "Passing model", fig.width = 6, fig.height = 4}
# get dataset without labels about completed passes
pass_data_unkown <- pass_data %>% select(-complete_pass) %>%
  data.matrix()

# get training labels
pass_success_labels <- pass_data %>%
  select(complete_pass) %>%
  data.matrix()

# put in the form xgboost understands
dtrain <- xgb.DMatrix(data = pass_data_unkown, label = pass_success_labels)

# define parameters
params <- list(
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  gamma = 0.14,
  subsample = 0.8,
  colsample_bytree = 0.7,
  min_child_weight = 4,
  max_delta_step = 9
)

set.seed(398)

# train the model
pass_train <- xgb.train(
  data = dtrain,
  objective = "binary:logistic",
  params = params,
  nrounds = 64,
  verbose = 2
)

# get information on how important each feature is
importance_matrix <- xgb.importance(names(pass_data_unkown), model = pass_train)

# plot feature importance
importance_matrix %>%
  ggplot(aes(reorder(Feature, Gain), Gain)) +
  geom_col(fill = "#99D9D9", color = "#001628") +
  coord_flip() +
  theme_bw() +
  labs(x = NULL, y = "Importance", caption = "data from Stathletes' Big Data Cup",
       title = "Pass completion probabilty feature importance",
       subtitle = "2019-2020 Erie Otters OHL games")

# prediction time

pass_data_predict <- predict(pass_train, pass_data_unkown) %>%
  as_tibble() %>%
  rename(completion_probability = value)

pass_data_predict <- bind_cols(pass_index, pass_data_predict)

pass_cv <- xgb.cv(
  data = dtrain,
  params = params,
  nthread = 4,
  nfold = 5,
  nrounds = 64,
  verbose = F,
  early_stopping_rounds = 25
)

print(
  paste0("CV test AUC: ",round(max(pass_cv$evaluation_log$test_auc_mean), 4))
)

```

The distance traveled by the intended pass is far and away the most important feature in this model. Strength state, which in this case is simply number of skaters for the passing team minus the number of skaters for the opposing team (ie a 5-on-4 powerplay yields a strength of 1). Whether a pass was direct or indirect (indirect being passes off the boards) was also more important than all of our pass classifications. The most significant classification came with our slot pass, which makes sense as it's generally more difficult to find open space right in front of the net to get a pass through^[According to the final model, passes across the slot averaged a 54.2% completion probability vs 73.7% for all other passes].

### Expected Goals

Now that we've got a baseline to determine if a pass will be completed or not, we need to quantify how likely it is that a completed pass will lead directly to a goal. Enter: a new expected goals model.

Expected goals models in the NHL are numerous, but one thing they all omit is passing data. Josh and Luke of [evolving-hockey.com](evolving-hockey.com) opted to include information from events preceding a shot^[https://rpubs.com/evolvingwild/395136/], which captures some of the pre-shot movement, in their model. But the raw NHL data doesn't include passing data. Fortunately for us, the Big Data Cup does include pass data.

Alex Novet has [shown in the past](https://hockey-graphs.com/2019/08/12/expected-goals-model-with-pre-shot-movement-part-1-the-model/)^[https://hockey-graphs.com/2019/08/12/expected-goals-model-with-pre-shot-movement-part-1-the-model/] that including passing data can indeed improve an expected goals model's accuracy, so we're going to lean into that quite a bit for this one.

```{r training xg model, echo = FALSE, warning = FALSE, fig.cap = "xG model", fig.width = 6, fig.height = 4}
shot_data <- bdc_data %>%
  separate(clock, into = c("minutes","seconds","milliseconds"), sep = ":", remove = FALSE) %>%
  mutate(
    minutes = as.numeric(minutes), seconds = as.numeric(seconds),
    period_seconds_remaining = (60 * minutes) + seconds,
    game_seconds_remaining = period_seconds_remaining + ((period - 1) * (20*60)),
    complete_pass = ifelse(event == "Play", 1, 0),
    home = ifelse(team == home_team, 1, 0),
    team_skaters = ifelse(home == 1, home_team_skaters, away_team_skaters),
    opponent_skaters = ifelse(home == 1, away_team_skaters, home_team_skaters),
    score_differential = ifelse(home == 1, home_team_goals - away_team_goals, away_team_goals - home_team_goals)
  ) %>%
  separate(home_team, into = c("home_city","home_nick"), remove = FALSE, sep = " ") %>%
  separate(away_team, into = c("away_city","away_nick"), remove = FALSE, sep = " ") %>%
  unite("game_id", c(game_date, away_nick, home_nick), sep = "_") %>%
    group_by(game_id, period) %>%
    mutate(
      event = ifelse(event == "Play", "Pass", event),
      event = ifelse(event == "Incomplete Play", "Incomplete Pass", event),
      next_event = lead(event),
      prev_event = lag(event),
      prev_event_2 = lag(prev_event),
      prev_event_3 = lag(prev_event_2),
      prev_event_4 = lag(prev_event_3),
      prev_event_5 = lag(prev_event_4),
      prev_period_seconds = lag(period_seconds_remaining),
      prev_period_seconds_2 = lag(prev_period_seconds),
      prev_period_seconds_3 = lag(prev_period_seconds_2),
      prev_period_seconds_4 = lag(prev_period_seconds_3),
      prev_period_seconds_5 = lag(prev_period_seconds_4),
      prev_x_coordinate = lag(x_coordinate),
      prev_x_coordinate_2 = lag(prev_x_coordinate),
      prev_x_coordinate_3 = lag(prev_x_coordinate_2),
      prev_x_coordinate_4 = lag(prev_x_coordinate_3),
      prev_x_coordinate_5 = lag(prev_x_coordinate_4),
      prev_y_coordinate = lag(y_coordinate),
      prev_y_coordinate_2 = lag(prev_y_coordinate),
      prev_y_coordinate_3 = lag(prev_y_coordinate_2),
      prev_y_coordinate_4 = lag(prev_y_coordinate_3),
      prev_y_coordinate_5 = lag(prev_y_coordinate_4),
      prev_x_coordinate_dest = lag(x_coordinate_2),
      prev_y_coordinate_dest = lag(y_coordinate_2)
    ) %>%
  ungroup()

shots <- shot_data %>%
  mutate(
    goal = ifelse(event == "Goal", 1, 0),
    event = ifelse(event == "Goal", "Shot", event)
  ) %>%
  filter(event == "Shot") %>%
  mutate(
    shot_type = detail_1,
    distance = abs(sqrt((x_coordinate - 189)^2 + (y_coordinate - 42.5)^2)),
    # shot angle
    theta = abs(atan((42.5-y_coordinate) / (189-x_coordinate)) * (180 / pi)),
    # fix behind the net angles
    theta = ifelse(x_coordinate > 189, 180 - theta, theta),
    traffic = detail_3,
    one_timer = detail_4,
    i5v4 = ifelse(team_skaters == 5 & opponent_skaters == 4, 1, 0),
    i5v3 = ifelse(team_skaters == 5 & opponent_skaters == 3, 1, 0),
    i6v5 = ifelse(team_skaters == 6 & opponent_skaters == 5, 1, 0),
    i6v4 = ifelse(team_skaters == 6 & opponent_skaters == 4, 1, 0),
    i5v5 = ifelse(team_skaters == 5 & opponent_skaters == 5, 1, 0),
    i4v4 = ifelse(team_skaters == 4 & opponent_skaters == 4, 1, 0),
    i3v3 = ifelse(team_skaters == 3 & opponent_skaters == 3, 1, 0),
    i4v5 = ifelse(team_skaters == 4 & opponent_skaters == 5, 1, 0),
    i3v5 = ifelse(team_skaters == 3 & opponent_skaters == 5, 1, 0),
    i5v6 = ifelse(team_skaters == 5 & opponent_skaters == 6, 1, 0),
    previous_passes = case_when(
      prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
        prev_event_4 == "Pass" & prev_event_5 == "Pass" ~ 5,
      prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
        prev_event_4 == "Pass" & prev_event_5 != "Pass" ~ 4,
      prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
        prev_event_4 != "Pass" ~ 3,
      prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 != "Pass" ~ 2,
      prev_event == "Pass" & prev_event_2 != "Pass" ~ 1,
      prev_event != "Pass" ~ 0
    ),
    time_since_pass = ifelse(
      prev_event == "Pass", prev_period_seconds - period_seconds_remaining, NA
    ),
    time_since_pass_2 = ifelse(
      # time since pass prior to last pass
      prev_event == "Pass" & prev_event_2 == "Pass",
      prev_period_seconds_2 - period_seconds_remaining,
      NA
    ),
    pass_distance = ifelse(
      prev_event == "Pass",
      abs(sqrt((prev_x_coordinate - prev_x_coordinate_dest)^2 + (prev_y_coordinate - prev_y_coordinate_dest)^2)),
      NA
    ),
    pass_behind_net = ifelse(
      prev_event == "Pass" & prev_x_coordinate > 189, 1, 0
    ),
    pass_slot = ifelse(
      prev_event == "Pass" &
        # 46 ft to top of the circles in the offensive zone
        prev_x_coordinate > 154 &
        # only count passes that cross from one side of the net to the other side
        ((prev_y_coordinate < 39.5 & prev_y_coordinate_dest > 42.5) |
        (prev_y_coordinate > 45.5 & prev_y_coordinate_dest < 42.5)),
      1, 0
    ),
    pass_stretch = ifelse(
      # passes from own defensive zone to someone within 5ft of offensive zone
      # arbitrary choice
      prev_event == "Pass" & prev_x_coordinate < 75 & prev_x_coordinate_dest > 120,
      1, 0
    ),
    # add shooter movement if shot location is different from pass destination
    shooter_movement_x = ifelse(
      prev_event == "Pass",
      x_coordinate - prev_x_coordinate_dest,
      NA
    ),
    shooter_movement_y = ifelse(
      prev_event == "Pass",
      y_coordinate - prev_y_coordinate_dest,
      NA
    ),
    shooter_movement = ifelse(
      prev_event == "Pass",
      abs(sqrt((shooter_movement_x)^2 + (shooter_movement_y)^2)),
      NA
    ),
    # add shot types
    deflection = ifelse(shot_type == "Deflection", 1, 0),
    fan = ifelse(shot_type == "Fan", 1, 0),
    slap = ifelse(shot_type == "Slapshot", 1, 0),
    snap = ifelse(shot_type == "Snapshot", 1, 0),
    wrap = ifelse(shot_type == "Wrap Around", 1, 0),
    wrist = ifelse(shot_type == "Wristshot", 1, 0),
    rebound = ifelse(prev_event == "Puck Recovery" & prev_event_2 == "Shot", 1, 0),
# note what type of play immediately preceded the shot
    prev_pass = 1 * (prev_event == "Pass"),
    prev_recovery = 1 * (prev_event == "Puck Recovery"),
    prev_shot = 1 * (prev_event == "Shot"),
    prev_takeaway = 1 * (prev_event == "Takeaway"),
    prev_zone_entry = 1 * (prev_event == "Zone Entry")
  )

shot_index <- select(shots, event_index)

# save this for later
sample_shots <- shots %>% filter(prev_pass == 1)

shots <- shots %>%
  select(-shot_type, -shooter_movement_x, -shooter_movement_y, -next_event) %>%
  select(
    score_differential, x_coordinate, y_coordinate, prev_x_coordinate, prev_y_coordinate,
    distance:prev_zone_entry, goal
  )

xg_model_feats <- shots %>% select(-goal) %>% names()

# prepare the data
goal_vector <- shots %>% select(goal) %>% data.matrix()
xg_train_data <- shots %>% select(-goal) %>% data.matrix()

shot_training_data <- xgb.DMatrix(
  data = xg_train_data,
  label = goal_vector
  )

# define parameters
params <- list(
  eval_metric = "auc",
  max_depth = 8,
  eta = 0.1,
  gamma = 1.5,
  subsample = 0.8,
  colsample_bytree = 0.7,
  min_child_weight = 2,
  max_delta_step = 9
)

set.seed(126)

xg_train <- xgb.train(
  data = shot_training_data,
  objective = "binary:logistic",
  params = params,
  nrounds = 95,
  verbose = 2
)

# get information on how important each feature is
importance_matrix <- xgb.importance(names(xg_train_data), model = xg_train)

# plot feature importance
importance_matrix %>%
  ggplot(aes(reorder(Feature, Gain), Gain)) +
  geom_col(fill = "#99D9D9", color = "#001628") +
  coord_flip() +
  theme_bw() +
  labs(x = NULL, y = "Importance", caption = "data from Stathletes' Big Data Cup",
       title = "Expected goal feature importance",
       subtitle = "2019-20 Erie Otters OHL games")

shots_predict <- predict(xg_train, data.matrix(select(shots, -goal))) %>%
  as_tibble() %>%
  rename(xg = value)

shots_predict <- bind_cols(shot_index, shots_predict)

# Cross Validation Results
set.seed(126)

xg_cv <- xgb.cv(
  data = shot_training_data,
  params = params,
  nthread = 4,
  nfold = 5,
  nrounds = 95,
  verbose = F,
  early_stopping_rounds = 25
)
print(
  paste0("CV test AUC: ",round(max(xg_cv$evaluation_log$test_auc_mean), 4))
)

```

The shooter's proximity to the net remains the most important feature, as is true with all other expected goals models in hockey that I can find. The angle to the net of the shot-taker is about as important as a new feature called "shooter movement." Shooter movement only has a value when a pass was immediately preceding the shot. It is definied as the distance between the shot location and the intended pass destination from the previous event.

Pass distance (for shots with a pass immediately prior) and whether or not the shot was a one-timer also prove to be important factors. As they aren't publicly available in full right now, these aren't normally included in public xGoal models.

The AUC (area under the curve) evaluation of the cross-validation results is better than that of the Evolving Wild xG model (0.828 vs EW's 0.782)^[https://rpubs.com/evolvingwild/395136/], which is encouraging in that it appears our new features are aiding the model's accuracy. But it's important to remember that this is only a 40-game sample and could be prone to over-fitting. Testing on future seasons would give us a better idea if this is the case.

We now have the two pieces we originally set out for when searching for xPA: pass completion probability and expected goals. By multiplying the pass completion probability by the expected goals value of the following shot, we can get an estimate of the likelihood of a primary assist being awarded to the passer. But what about passes to dangerous areas that are, for whatever reason, incomplete and don't have a shot recorded afterwards? Or passes that are completed but the recipient decides not to shoot?

We can account for these cases by calculating the expected goals value of *theoretical* shots taken after a pass. Our data just needs some small adjustments, and some educated guesses. We can adjust most of the variables fairly easily to account for the new phantom shot, but five others require some extra work:

  *   shot type
  *   traffic
  *   one-timer
  *   shooter movement
  *   time since pass
  
Shot type, traffic, one-timer, and time since pass were all selected at random, weighed by how often they occured in our full dataset of actual shots taken. Shooter movement was sampled from all the shots within one standard deviation of the mean shooter movement in our shots data, and adjusted to ensure all shot locations remained within the bounds of the ice sheet.

The variables were randomly selected and the expected goals calculated 500 times, and with the average xG value being taken for each phantom shot.

```{r phantom shots, include = FALSE, warning = FALSE}
# combining xg + xc into full data set

bdc_data <- bdc_data %>%
  left_join(shots_predict, by = "event_index") %>%
  left_join(pass_data_predict, by = "event_index")

# let's restrict our "phantom" shooter movements to within one SD of the mean
# in both directions
sample_movement <- sample_shots %>%
  filter(
    between(
      shooter_movement_x,
      mean(sample_shots$shooter_movement_x) - sd(sample_shots$shooter_movement_x),
      mean(sample_shots$shooter_movement_x) + sd(sample_shots$shooter_movement_x)
            ) &
    between(
      shooter_movement_y,
      mean(sample_shots$shooter_movement_y) - sd(sample_shots$shooter_movement_y),
      mean(sample_shots$shooter_movement_y) + sd(sample_shots$shooter_movement_y)
    )
         )

# only cut out 800 of 2800 shots

# creating phantom shots
inc_shots_predict <- NULL

set.seed(487)

for(i in 1:500){
  tictoc::tic()
  print(paste0("Getting randomized shot inputs, ",i," of 5..."))
  
  inc_shots <- shot_data %>%
    filter(
      event == "Incomplete Pass" |
      #  add completed passes that didn't lead directly to shot
      #  we'll make the recipient shoot now
      (event == "Pass" & next_event != "Shot")
      ) %>%
    # filter to just passes heading past center ice
    filter(x_coordinate_2 > 100) %>%
    mutate(
      distance = # distance to net from intended pass destination
        abs(sqrt((x_coordinate_2 - 189)^2 + (y_coordinate_2 - 42.5)^2)),
      theta = # from intended pass destination
        abs(atan((42.5 - y_coordinate_2) / (189 - x_coordinate_2)) * (180 / pi)), # from evolvingwild
      theta = ifelse(x_coordinate_2 > 189, 180 - theta, theta), # fix behind the net angles
      i5v4 = ifelse(team_skaters == 5 & opponent_skaters == 4, 1, 0),
      i5v3 = ifelse(team_skaters == 5 & opponent_skaters == 3, 1, 0),
      i6v5 = ifelse(team_skaters == 6 & opponent_skaters == 5, 1, 0),
      i6v4 = ifelse(team_skaters == 6 & opponent_skaters == 4, 1, 0),
      i5v5 = ifelse(team_skaters == 5 & opponent_skaters == 5, 1, 0),
      i4v4 = ifelse(team_skaters == 4 & opponent_skaters == 4, 1, 0),
      i3v3 = ifelse(team_skaters == 3 & opponent_skaters == 3, 1, 0),
      i4v5 = ifelse(team_skaters == 4 & opponent_skaters == 5, 1, 0),
      i3v5 = ifelse(team_skaters == 3 & opponent_skaters == 5, 1, 0),
      i5v6 = ifelse(team_skaters == 5 & opponent_skaters == 6, 1, 0),
      # move previous events back one spot because the new event is a shot
      prev_event_5 = prev_event_4,
      prev_event_4 = prev_event_3,
      prev_event_3 = prev_event_2,
      prev_event_2 = prev_event,
      prev_event = "Pass",
      previous_passes = case_when(
        # this should be fine since we've already moved the events back a spot
        prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
          prev_event_4 == "Pass" & prev_event_5 == "Pass" ~ 5,
        prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
          prev_event_4 == "Pass" & prev_event_5 != "Pass" ~ 4,
        prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 == "Pass" &
          prev_event_4 != "Pass" ~ 3,
        prev_event == "Pass" & prev_event_2 == "Pass" & prev_event_3 != "Pass" ~ 2,
        prev_event == "Pass" & prev_event_2 != "Pass" ~ 1,
        prev_event != "Pass" ~ 0
      ),
      # adding in random shooter movement
      shooter_movement_x = sample(sample_movement$shooter_movement_x, 1),
      # fix selections resulting in shooters outside the ice surface (>200ft)
      shooter_movement_x = ifelse(x_coordinate_2 + shooter_movement_x >= 200,
                                  199 - x_coordinate_2,
                                  shooter_movement_x),
      # adding in random shooter movement
      shooter_movement_y = sample(sample_movement$shooter_movement_y, 1),
      # fix selections resulting in shooters outside the ice surface (>85ft)
      shooter_movement_y = ifelse(y_coordinate_2 + shooter_movement_y >= 85,
                                  88 - y_coordinate_2,
                                  shooter_movement_y),
      # same but for the right side of the ice (<0ft)
      shooter_movement_y = ifelse(y_coordinate_2 + shooter_movement_y <= 0,
                                  1 - y_coordinate_2,
                                  shooter_movement_y),
      # adjust previous coordinates back a spot
      prev_x_coordinate_5 = prev_x_coordinate_4,
      prev_x_coordinate_4 = prev_x_coordinate_3,
      prev_x_coordinate_3 = prev_x_coordinate_2,
      prev_x_coordinate_2 = prev_x_coordinate,
      prev_x_coordinate = x_coordinate,
      prev_x_coordinate_dest = x_coordinate_2,
      x_coordinate = x_coordinate_2 + shooter_movement_x,
      prev_y_coordinate_5 = prev_y_coordinate_4,
      prev_y_coordinate_4 = prev_y_coordinate_3,
      prev_y_coordinate_3 = prev_y_coordinate_2,
      prev_y_coordinate_2 = prev_y_coordinate,
      prev_y_coordinate = y_coordinate,
      prev_y_coordinate_dest = y_coordinate_2,
      y_coordinate = y_coordinate_2 + shooter_movement_y,
      # adjust the time remaining back a spot
      prev_period_seconds_2 = prev_period_seconds,
      prev_period_seconds = period_seconds_remaining,
      time_since_pass =
        # random sample of time elapsed between pass and shot
        sample(sample_shots$time_since_pass, 1),
      period_seconds_remaining = prev_period_seconds + time_since_pass,
      time_since_pass_2 = ifelse(
        prev_event == "Pass" & prev_event_2 == "Pass",
        prev_period_seconds_2 - period_seconds_remaining,
        NA
      ),
      pass_distance = 
        abs(sqrt((prev_x_coordinate - prev_x_coordinate_dest)^2 +
                   (prev_y_coordinate - prev_y_coordinate_dest)^2)
        ),
      pass_behind_net = ifelse(
        prev_event == "Pass" & prev_x_coordinate > 189, 1, 0
      ),
      pass_slot = ifelse(
        prev_event == "Pass" &
          # 46 ft to top of the circles in the offensive zone
          prev_x_coordinate > 154 &
          # only count passes that cross from one side of the net to the other side
          ((prev_y_coordinate < 42.5 & prev_y_coordinate_dest > 42.5) |
             (prev_y_coordinate > 42.5 & prev_y_coordinate_dest < 42.5)),
        1, 0
      ),
      pass_stretch = ifelse(
        # passes from own defensive zone to someone within 5ft of offensive zone
        # arbitrary choice
        prev_event == "Pass" & prev_x_coordinate < 75 & prev_x_coordinate_dest > 120,
        1, 0
      ),
      shooter_movement =
        abs(sqrt(shooter_movement_x^2 + shooter_movement_y^2)),
      # shot types: weighted random selections of all shots
      shot_type = sample(
        sample_shots$shot_type, 1
      ),
      deflection = ifelse(shot_type == "Deflection", 1, 0),
      fan = ifelse(shot_type == "Fan", 1, 0),
      slap = ifelse(shot_type == "Slapshot", 1, 0),
      snap = ifelse(shot_type == "Snapshot", 1, 0),
      wrap = ifelse(shot_type == "Wrap Around", 1, 0),
      wrist = ifelse(shot_type == "Wristshot", 1, 0),
      traffic = sample(sample_shots$traffic, 1),
      one_timer = sample(sample_shots$one_timer, 1),
      rebound = ifelse(prev_event == "Puck Recovery" & prev_event_2 == "Shot", 1, 0),
      # previous event is always pass in these cases
      prev_pass = 1,
      prev_recovery = 0,
      prev_shot = 0,
      prev_takeaway = 0,
      prev_zone_entry = 0
    )
  
  inc_shot_index <- select(inc_shots, event_index)
  
  inc_shots <- inc_shots %>%
    select(
      all_of(xg_model_feats)
    )
  
  inc_shots_predict_i <- predict(xg_train, data.matrix(inc_shots)) %>%
    as_tibble() %>%
    rename(xg2 = value)
  
  inc_shots_predict_i <- bind_cols(inc_shot_index, inc_shots_predict_i) %>%
    pivot_wider(1:2, names_from = event_index, values_from = xg2)
  
  inc_shots_predict <- bind_rows(inc_shots_predict, inc_shots_predict_i)
  rm(inc_shots_predict_i)
  tictoc::toc()
}

colMax <- function(data) sapply(data, max, na.rm = TRUE)
colMin <- function(data) sapply(data, min, na.rm = TRUE)

inc_shots_predict <- inc_shots_predict %>%
  colMeans() %>%
  as_tibble() %>%
  rename(mean_xg = value) %>%
  bind_cols(
    (inc_shots_predict %>%
       colMax() %>%
       as_tibble() %>%
       rename(max_xg = value)),
    (inc_shots_predict %>%
       colMin() %>%
       as_tibble() %>%
       rename(min_xg = value)),
    inc_shot_index
    ) %>%
  select(event_index, everything())

bdc_data <- bdc_data %>%
  left_join(inc_shots_predict, by = "event_index")

bdc_data <- bdc_data %>%
  separate(clock, into = c("minutes","seconds","milliseconds"), sep = ":", remove = FALSE) %>%
  mutate(
    minutes = as.numeric(minutes), seconds = as.numeric(seconds),
    period_seconds_remaining = (60 * minutes) + seconds,
    game_seconds_remaining = period_seconds_remaining + ((period - 1) * (20*60)),
    complete_pass = ifelse(event == "Play", 1, 0),
    home = ifelse(team == home_team, 1, 0),
    team_skaters = ifelse(home == 1, home_team_skaters, away_team_skaters),
    opponent_skaters = ifelse(home == 1, away_team_skaters, home_team_skaters),
    strength = team_skaters - opponent_skaters,
    score_differential = ifelse(home == 1, home_team_goals - away_team_goals, away_team_goals - home_team_goals)
  ) %>%
  separate(home_team, into = c("home_city","home_nick"), remove = FALSE, sep = " ") %>%
  separate(away_team, into = c("away_city","away_nick"), remove = FALSE, sep = " ") %>%
  unite("game_id", c(game_date, away_nick, home_nick), sep = "_") %>%
  group_by(game_id, period) %>%
  mutate(
    next_xg = ifelse(event == "Incomplete Play", mean_xg, lead(xg))
  ) %>%
  ungroup() %>%
  mutate(
    xa = completion_probability * next_xg
  )

scoring_events <- c("Play","Incomplete Play","Shot","Goal")

bdc_data <- bdc_data %>%
  mutate(
    goal = ifelse(event == "Goal", 1, 0),
    pass = ifelse(event == "Play" | event == "Incomplete Play", 1, 0)
  )

otters_stats <- read_csv(url("https://raw.githubusercontent.com/danmorse314/bdc21/main/otters%20game%20stats.csv"),
                         col_types = cols()) %>%
  left_join(
    read_csv(
      url(
        "https://raw.githubusercontent.com/danmorse314/bdc21/main/otters%20positions.csv"
      ),
      col_types = cols(),
    ),
    by = "player"
  )

# get just 5v5 stats
otters_stats_ev <- read_csv(url("https://raw.githubusercontent.com/danmorse314/bdc21/main/otters%20ev%20stats.csv"),
                         col_types = cols()) %>%
  left_join(
    read_csv(
      url(
        "https://raw.githubusercontent.com/danmorse314/bdc21/main/otters%20positions.csv"
      ),
      col_types = cols(),
    ),
    by = "player"
  )

box <- bdc_data %>%
  mutate(shot = ifelse(event == "Shot" | event == "Goal", 1 ,0)) %>%
  filter(
    event %in% scoring_events
  ) %>%
  group_by(player) %>%
  summarize(
    goals = sum(goal),
    xg = round(sum(xg, na.rm = TRUE),1),
    xpa = round(sum(xa, na.rm = TRUE),1),
    shots = sum(shot),
    passes = sum(pass),
    cp = round(sum(complete_pass, na.rm = TRUE) / passes * 100, 1),
    xcp = round(mean(completion_probability, na.rm = TRUE) * 100, 1),
    cpox = cp - xcp,
    .groups = "drop"
  ) %>%
  ungroup() %>%
  arrange(-xpa) %>%
  left_join(select(otters_stats, -goals), by = "player") %>%
  mutate(
    gox = goals - xg,
    aox = assists_primary - xpa,
    xg_per_10 = round(xg / shots * 10, 2),
    assists = assists_primary + assists_secondary,
    xpa_per_100 = round(xpa / passes * 100, 2)
  ) %>%
  select(player, position, goals, xg, gox, assists_primary, xpa, xpa_per_100,
         xg_per_10, aox, cp, xcp, cpox, assists_secondary, assists, passes, shots) %>%
  arrange(-xpa_per_100) %>%
  filter(!is.na(position) & position != "G")

box_ev <- bdc_data %>%
  filter(team_skaters == 5 & opponent_skaters == 5) %>%
  mutate(shot = ifelse(event == "Shot" | event == "Goal", 1 ,0)) %>%
   filter(
     event %in% scoring_events
   ) %>%
   group_by(player) %>%
   summarize(
     goals = sum(goal),
     xg = round(sum(xg, na.rm = TRUE),1),
     xpa = round(sum(xa, na.rm = TRUE),1),
     shots = sum(shot),
     passes = sum(pass),
     cp = round(sum(complete_pass, na.rm = TRUE) / passes * 100, 1),
     xcp = round(mean(completion_probability, na.rm = TRUE) * 100, 1),
     cpox = cp - xcp,
     .groups = "drop"
   ) %>%
   ungroup() %>%
   arrange(-xpa) %>%
   left_join(select(otters_stats_ev, -goals), by = "player") %>%
   mutate(
     gox = goals - xg,
     aox = assists_primary - xpa,
     xg_per_10 = round(xg / shots * 10, 2),
     assists = assists_primary + assists_secondary,
     xpa_per_100 = round(xpa / passes * 100, 2)
   ) %>%
  select(player, position, goals, xg, gox, assists_primary, xpa, xpa_per_100,
         xg_per_10, aox, cp, xcp, cpox, assists_secondary, assists, passes, shots) %>%
  arrange(-xpa_per_100) %>%
  filter(!is.na(position) & position != "G")
```

##    Results

With that all taken care of, we can now calculate the expected goals and assists of every pass & shot in this data, and compare it to the goals, primary assists, and secondary assists scored on the OHL's [official website](https://ontariohockeyleague.com/stats)^[https://ontariohockeyleague.com/stats]. I've narrowed it down to 5-on-5 stats as that's when the majority of the game is played, and other situations change the play style dramatically.

```{r player stats, echo = FALSE, warning = FALSE}
# even strength stats
box_ev %>%
  filter(passes >= 50) %>%
  select(player, position, goals, xg, assists_primary, xpa, aox, xpa_per_100, passes, cpox) %>%
  gt(rowname_col = "player") %>%
  tab_header(
    title = "Erie Otters Advanced Stats",
    subtitle = "select 2019-20 games, min. 50 passes, 5-on-5 situations"
  ) %>%
  tab_row_group(
    group = "Forwards",
    rows = (position == "LW" | position == "C" | position == "RW")
  ) %>%
  tab_row_group(
    group = "Defense",
    rows = position == "D"
  ) %>%
  cols_label(
    player = md("**Name**"),
    position = md("**Position**"),
    goals = md("**Goals**"),
    xg = md("**xG**"),
    #gox = md("**GOX**"),
    assists_primary = md("**PA**"),
    xpa = md("**xPA**"),
    xpa_per_100 = md("**xPA/100**"),
    aox = md("**PAOx**"),
    #cp = md("**Pass Comp%**"),
    #xcp = md("**Expected Pass Comp%**"),
    cpox = md("**CPOx**"),
    #assists_secondary = md("**A2**"),
    #assists = md("**Assists**"),
    passes = md("**Passes**")
  ) %>%
  cols_align(
    align = "center"
  )
```

It looks like we're measuring something close to primary assists! A linear model shows that xPA explains around 57% of the variance in an individual player's primary assist total at 5-on-5 and around 77% in all situations.

One of the more noticable differences is in how stark a contrast there is between forwards and defensemen. Taking each player's expected primary assists per 100 passes made (xPA/100), the best defenseman on the team is still a less dangerous passer than the worst forward on the team. That makes sense considering defensemen are making a significant portion of their passes on the perimiter of the offensive zone, where the expected goals are low, while forwards send more passes to the middle or down low.

On a contextual level, xPA/100 is measuring how "dangerous" a player's passes were on average.

Among forwards on the Otters last year, Hayden Fowler stands out as someone who might warrant more attention. His xPA/100 was seconds on the team, but he only came away with 4 primary assists. The discrepency makes sense when you see his pass completion percentage over expectation (CPOx) is -8.4%. If we assume that will regress towards the mean next year, we could view Fowler as an under-the-radar player set for a breakout playmaker season. However, there is the chance that CPOx is a skill-based trait, which would mean that while he has an aggressive mindset when it comes to sending passes into dangerous areas, he's simply not very good at actually making the pass. Measuring the season-to-season correlation of this statistic in the future would help give us a better understanding of what to expect in future years.

Contextually, we can also use this as a stylistic measure. Plotting the xPA rate and the xG rate gives us a good idea of which players prefer to shoot from dangerous areas and which prefer to pass.

```{r player styles, echo = FALSE, fig.cap = "Player styles", fig.width = 6, fig.height = 4}
box_ev <- box_ev %>%
  mutate(position_group = ifelse(position == "D", "Defense", "Forwards"))

f.xg <- mean(filter(box_ev, position_group == "Forwards")$xg_per_10)
f.xpa <- mean(filter(box_ev, position_group == "Forwards")$xpa_per_100)
d.xg <- mean(filter(box_ev, position_group != "Forwards")$xg_per_10)
d.xpa <- mean(filter(box_ev, position_group != "Forwards")$xpa_per_100)

def.plot <- box_ev %>%
  filter(passes >= 50) %>%
  filter(position_group == "Defense") %>%
  ggplot(aes(xpa_per_100, xg_per_10)) +
  annotate("text", x = .8, y = .9, hjust = 0, vjust = 0,
           label = "Shoot\nfirst", color = "blue") +
  annotate("text", x = .8, y = .3, hjust = 0, vjust = 0,
           label = "NA", color = "blue") +
  annotate("text", x = 1.7, y = .9, hjust = 1, vjust = 0,
           label = "Balanced\nattack", color = "blue") +
  annotate("text", x = 1.7, y = .3, hjust = 1, vjust = 0,
           label = "Pass\nfirst", color = "blue") +
  geom_hline(yintercept = f.xg,
             linetype = "dashed", color = "red") +
  geom_vline(xintercept = f.xpa,
             linetype = "dashed", color = "red") +
  geom_label(aes(label = player), alpha = .5) +
  theme_bw() +
  scale_x_continuous(breaks = scales::pretty_breaks(), limits = c(0.8,1.7)) +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  labs(title = "Erie Otters play styles", x = "xPA/100 passes", y = "xG/10 shots",
       subtitle = "Defense, min. 100 passes, 5-on-5 situations",
       caption = "chart: @danmorse_ | data: Stathletes")

fwd.plot <- box_ev %>%
  filter(passes >= 50) %>%
  filter(position_group == "Forwards") %>%
  ggplot(aes(xpa_per_100, xg_per_10)) +
  annotate("text", x = .8, y = .9, hjust = 0, vjust = 0,
           label = "Shoot\nfirst", color = "blue") +
  annotate("text", x = .8, y = .3, hjust = 0, vjust = 0,
           label = "NA", color = "blue") +
  annotate("text", x = 1.7, y = .9, hjust = 1, vjust = 0,
           label = "Balanced\nattack", color = "blue") +
  annotate("text", x = 1.7, y = .3, hjust = 1, vjust = 0,
           label = "Pass\nfirst", color = "blue") +
  geom_hline(yintercept = f.xg,
             linetype = "dashed", color = "red") +
  geom_vline(xintercept = f.xpa,
             linetype = "dashed", color = "red") +
  geom_label(aes(label = player), alpha = .5) +
  theme_bw() +
  scale_x_continuous(breaks = scales::pretty_breaks(), limits = c(0.8,1.7)) +
  scale_y_continuous(breaks = scales::pretty_breaks()) +
  labs(title = "Erie Otters play styles", x = "xPA/100 passes", y = "xG/10 shots",
       subtitle = "Forwards, min. 100 passes, 5-on-5 situations",
       caption = "chart: @danmorse_ | data: Stathletes")

fwd.plot
```

The Otters' leading goal scorer (both in expected and observed goals) was Chad Yetman. Yetman was also the 2nd least likely forward to pass into dangerous areas on the team. Meanwhile, his frequent linemate Maxim Golod was on the opposite end of the spectrum, opting to pass more often than all but one other forward on the team. The pair playing on the same line produced the most goals on the team last year, with Golod netting the primary assist on nearly half of Yetman's goals at even strength.

```{r yetman alluvial plot, warning = FALSE, echo = FALSE, fig.cap = "Chad Yetman's goals", fig.width = 6, fig.height = 4}

yetman_goals <- read_csv(url("https://raw.githubusercontent.com/danmorse314/bdc21/main/otters%20goal%20details.csv"),
                         col_types = cols()) %>%
  filter(strength == "ev") %>%
  select(a2, a1, g) %>%
  filter(g == "Chad Yetman") %>%
  group_by(a2, a1, g) %>%
  summarize(n = n(), .groups = "drop") %>%
  ungroup()

ggplot(yetman_goals,
       aes(y = n, axis1 = a2, axis2 = a1, axis3 = g)) +
  ggalluvial::geom_alluvium(width = 1/12, aes(fill = a1), show.legend = FALSE) +
  ggalluvial::geom_stratum(width = 1/12, fill = "black", color = "grey") +
  geom_label(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_continuous(breaks = 1:3, labels = c("A2","A1","G"), limits = c(.8,3.2)) +
  scale_fill_brewer(type = "qual", palette = "Set1") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        plot.caption = element_text(hjust = .9),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        axis.text.y = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_blank()) +
  labs(title = "Maxim Golod fed Chad Yetman last year",
       subtitle = "2019-20 Erie Otters games, 5-on-5 situations",
       caption = "data from Stathletes")

print(paste0("Yetman goals with Golod as A1: ",
  round(
    yetman_goals %>%
    group_by(a1) %>%
    summarize(n = sum(n)) %>%
    filter(a1 == "Maxim Golod") %>%
    pull(n) / sum(yetman_goals$n) * 100, 1
  ), "%"
))

```

Coaches frequently attempt to make forward lines with good chemistry. A breakdown like one in Fig. 3 could aid in that search, as we've already seen that pairing a player whose game is favored by xPA with a player who looks great by xG can lead to a winning combination. Based on this data, the play styles of Hayden Fowler and Elias Cohen should mesh very well together. Giving them more playing time together should lead to more high-danger shots by Cohen on high-danger passes from Fowler.

In the future, the passing model could use more fine tuning. Perhaps adding a new feature like whether or not the pass was sent through traffic would give it a boost. As of now, the xPA model can already be helpful in evaluating different player styles. With year-over-year testing, it will also benefit player evaluation once we determine which elements are skill-based and which are likely to regress towards the mean.
