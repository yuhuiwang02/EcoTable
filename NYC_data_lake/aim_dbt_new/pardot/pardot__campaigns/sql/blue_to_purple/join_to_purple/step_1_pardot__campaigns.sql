with campaigns as (

    select *
    from "pardot"."public_stg_pardot"."stg_pardot__campaign"

), prospects as (

    select *
    from "pardot"."public_stg_pardot"."stg_pardot__prospect"

), opportunities as (

    select *
    from "pardot"."public_pardot"."int__opportunities_by_campaign"

), prospects_xf as (

    select 
        campaign_id,
        count(*) as count_prospects 
    from prospects
    group by 1

), joined as (

    select *
    from campaigns
    left join opportunities
        using (campaign_id)
    left join prospects_xf
        using (campaign_id)

)   

select *
from joined