with guide_history as (

    select *
    from "pendo"."public_stg_pendo"."stg_pendo__guide_history"

),

latest_guide as (
    select
      *,
      row_number() over(partition by guide_id order by last_updated_at desc) as latest_guide_index
    from guide_history
)

select *
from latest_guide
where latest_guide_index = 1