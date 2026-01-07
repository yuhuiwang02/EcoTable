with feature_history as (

    select *
    from "pendo"."public_stg_pendo"."stg_pendo__feature_history"

),

latest_feature as (
    select
      *,
      row_number() over(partition by feature_id order by last_updated_at desc) as latest_feature_index
    from feature_history
)

select *
from latest_feature
where latest_feature_index = 1