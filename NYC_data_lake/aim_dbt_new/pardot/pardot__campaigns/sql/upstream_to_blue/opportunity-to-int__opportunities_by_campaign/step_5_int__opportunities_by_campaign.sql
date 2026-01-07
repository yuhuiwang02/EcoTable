

with opportunities as (

    select *
    from "pardot"."public_pardot"."int__opportunity_tmp"

), aggregated as (

    select 
        campaign_id,
        count(*) as count_opportunities,

        
        count(case when opportunity_status = 'Won' then 1 end) as count_opportunities_won,
        sum(case when opportunity_status = 'Won' then amount end) as sum_opportunity_amount_won
         , 
        
        count(case when opportunity_status = 'Open' then 1 end) as count_opportunities_open,
        sum(case when opportunity_status = 'Open' then amount end) as sum_opportunity_amount_open
         , 
        
        count(case when opportunity_status = 'Lost' then 1 end) as count_opportunities_lost,
        sum(case when opportunity_status = 'Lost' then amount end) as sum_opportunity_amount_lost
        
        
    
    from opportunities
    group by 1

)

select *
from aggregated