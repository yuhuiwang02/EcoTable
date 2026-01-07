

with base as (

    select * 
    from "google_ads"."public_google_ads_dev"."stg_google_ads__search_term_keyword_stats_tmp"
),

fields as (

    select
        
    cast(null as TEXT) as 
    
    _fivetran_id
    
 , 
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as integer) as 
    
    ad_group_id
    
 , 
    cast(null as integer) as 
    
    campaign_id
    
 , 
    cast(null as integer) as 
    
    clicks
    
 , 
    cast(null as float) as 
    
    conversions
    
 , 
    cast(null as float) as 
    
    conversions_value
    
 , 
    cast(null as integer) as 
    
    cost_micros
    
 , 
    cast(null as integer) as 
    
    customer_id
    
 , 
    cast(null as date) as 
    
    date
    
 , 
    cast(null as integer) as 
    
    impressions
    
 , 
    cast(null as TEXT) as 
    
    info_text
    
 , 
    cast(null as TEXT) as 
    
    keyword_ad_group_criterion
    
 , 
    cast(null as TEXT) as 
    
    search_term
    
 , 
    cast(null as TEXT) as 
    
    search_term_match_type
    
 , 
    cast(null as TEXT) as 
    
    status
    
 , 
    cast(null as integer) as 
    
    view_through_conversions
    
 


        


, cast('' as TEXT) as source_relation



    from base
),

final as (
    
    select 
        source_relation, 
        customer_id as account_id,
        date as date_day,
        _fivetran_id as search_term_id,
        cast(ad_group_id as TEXT) as ad_group_id,
        campaign_id,
        keyword_ad_group_criterion,
        
        

  
    

    split_part(
        keyword_ad_group_criterion,
        '~',
        2
        )


  

 as criterion_id,
        search_term,
        info_text as keyword_text,
        search_term_match_type,
        status, 
        coalesce(clicks, 0) as clicks, 
        coalesce(cost_micros, 0) / 1000000.0 as spend, 
        coalesce(impressions, 0) as impressions,
        coalesce(conversions, 0) as conversions,
        coalesce(conversions_value, 0) as conversions_value,
        coalesce(view_through_conversions, 0) as view_through_conversions

        





    from fields
)

select *
from final