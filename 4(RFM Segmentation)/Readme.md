1) Load & clean

Read CSVs into DataFrames: users, friendships, interactions, posts.

Convert time columns to true datetimes:

interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'])
posts['Timestamp'] = pd.to_datetime(posts['Timestamp'])

2) Tie interactions to the posts they reference

Merge interactions ↔ posts on Post_ID:

interactions_posts = interactions.merge(posts, on='Post_ID', how='left')


Rename columns so it’s clear who is who and which time is which:

Username_x → Interacting_Username (who did the like/comment/share)

Username_y → Post_Author

Timestamp_x → Interaction_Timestamp

Timestamp_y → Post_Timestamp

Result: each row = one interaction, with the post text/author/time attached.

3) Add demographics of the actor (and optionally the author)

Merge in the interactor’s demographics from users:

interactions_full = interactions_posts.merge(
    users, left_on='Interacting_Username', right_on='Username', how='left'
)


Also merge in the post author’s demographics (2nd merge with suffixes):

interactions_full = interactions_full.merge(
    users, left_on='Post_Author', right_on='Username', how='left',
    suffixes=('_Interactor', '_Author')
)


Result: columns like Age_Interactor, Gender_Author, etc.

4) Flag whether the interactor and author are friends

Mark all friendships rows with is_friend = 1.

Merge friendships onto interactions_full using pair (Interacting_Username, Post_Author):

interactions_full = interactions_full.merge(
    friendships,
    left_on=['Interacting_Username','Post_Author'],
    right_on=['Username','Friend'],
    how='left'
)
interactions_full['is_friend'] = interactions_full['is_friend'].fillna(0).astype(int)


Result: is_friend = 1 if that specific interactor–author pair exists in friendships, else 0.

Note: This treats friendship as directional. If your friendships are undirected, you likely want to add the reversed pairs too (e.g., append a copy of friendships with columns swapped) before merging.

5) Prepare RFM inputs

Ensure Interaction_Timestamp is datetime (again).

Assign “Monetary” points to interaction types:

interaction_values = {'like':1, 'comment':2, 'share':3}
interactions_full['Monetary_Value'] = interactions_full['Interaction_Type'].map(interaction_values)

6) Compute the RFM table (per interacting user)

Reference date = latest interaction in the dataset:

max_date = interactions_full['Interaction_Timestamp'].max()


Aggregate to RFM:

Recency: days since each user’s most recent interaction (smaller is better).

Frequency: count of interactions.

Monetary: sum of the mapped weights (like=1, comment=2, share=3).

rfm = interactions_full.groupby('Interacting_Username').agg(
    Recency=('Interaction_Timestamp', lambda x: (max_date - x.max()).days),
    Frequency=('Interaction_Type','count'),
    Monetary=('Monetary_Value','sum')
).reset_index()


Result: one row per interactor with Recency/Frequency/Monetary metrics.

7) Score & segment users by RFM

Quartile scores (1–4) for each metric:

Recency is reversed (more recent ⇒ higher score): labels=[4,3,2,1].

Frequency/Monetary normal (more ⇒ higher): labels=[1,2,3,4].

Concatenate to form RFM_Score like “144”, “333”, etc.

Convert to numeric and segment into Low / Middle / High by overall score quartiles.

Result: rfm now has R_Quartile, F_Quartile, M_Quartile, RFM_Score, RFM_Group.

8) Relate RFM to demographics (example with age)

Attach interactor age to the rfm table and compute summary stats per group:

user_ages = interactions_full[['Interacting_Username','Age_Interactor']].drop_duplicates()
rfm_age = rfm.merge(user_ages, on='Interacting_Username', how='left')
age_stats = rfm_age.groupby('RFM_Group')['Age_Interactor'].agg(['mean','median','min','max','count'])


Result: age_stats shows how age varies across Low/Middle/High RFM groups.
