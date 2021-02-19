library(tidyverse)
library(xgboost, exclude = "slice")
library(gt)
library(ggalluvial)

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

# reverse %in%
`%not_in%` <- purrr::negate(`%in%`)

# getting rid of strength states because it's too cluttered
ignore_feats <- grep("^[iI].*",importance_matrix$Feature, value = TRUE)

importance_matrix <- importance_matrix %>%
  filter(Feature %not_in% ignore_feats)

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
  annotate("text", x = .2, y = .26, hjust = 0, vjust = 1,
           label = "Shoot\nfirst", color = "blue") +
  annotate("text", x = .2, y = 0.1, hjust = 0, vjust = 0,
           label = "NA", color = "blue") +
  annotate("text", x = .8, y = .26, hjust = 1, vjust = 1,
           label = "Balanced\nattack", color = "blue") +
  annotate("text", x = .8, y = .1, hjust = 1, vjust = 0,
           label = "Pass\nfirst", color = "blue") +
  geom_hline(yintercept = d.xg,
             linetype = "dashed", color = "red") +
  geom_vline(xintercept = d.xpa,
             linetype = "dashed", color = "red") +
  geom_label(aes(label = player)) +
  theme_bw() +
  scale_x_continuous(breaks = scales::pretty_breaks(), limits = c(0.15,0.8)) +
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

def.plot

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
