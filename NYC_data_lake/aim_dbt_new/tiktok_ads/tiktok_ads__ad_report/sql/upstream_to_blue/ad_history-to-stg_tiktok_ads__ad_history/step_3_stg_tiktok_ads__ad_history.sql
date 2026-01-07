

with base as (

    select *
    from "tiktok_ads"."public_tiktok_ads_dev"."stg_tiktok_ads__ad_history_tmp"
), 

fields as (

    select
        
    
    
    ad_id
    
 as 
    
    ad_id
    
, 
    
    
    ad_name
    
 as 
    
    ad_name
    
, 
    
    
    adgroup_id
    
 as 
    
    adgroup_id
    
, 
    
    
    advertiser_id
    
 as 
    
    advertiser_id
    
, 
    
    
    call_to_action
    
 as 
    
    call_to_action
    
, 
    
    
    campaign_id
    
 as 
    
    campaign_id
    
, 
    
    
    click_tracking_url
    
 as 
    
    click_tracking_url
    
, 
    
    
    impression_tracking_url
    
 as 
    
    impression_tracking_url
    
, 
    
    
    landing_page_url
    
 as 
    
    landing_page_url
    
, 
    
    
    updated_at
    
 as 
    
    updated_at
    




    
        


, cast('' as TEXT) as source_relation




    from base
), 

final as (

    select
        source_relation,  
        ad_id,
        cast(updated_at as timestamp) as updated_at,
        adgroup_id as ad_group_id,
        advertiser_id,
        campaign_id,
        ad_name,
        call_to_action,
        click_tracking_url,
        impression_tracking_url,
        

  
    

    split_part(
        landing_page_url,
        '?',
        1
        )


  

 as base_url,
        
    
    cast(

  
    

    split_part(
        

  
    

    split_part(
        

    replace(
        

    replace(
        

    replace(
        landing_page_url,
        'android-app://',
        ''
    )


,
        'http://',
        ''
    )


,
        'https://',
        ''
    )


,
        '/',
        1
        )


  

,
        '?',
        1
        )


  

 as TEXT)
 as url_host,
        '/' || 
    
    cast(

  
    

    split_part(
        

    right(
        

    replace(
        

    replace(
        landing_page_url,
        'http://',
        ''
    )


,
        'https://',
        ''
    )


,
        

    length(
        

    replace(
        

    replace(
        landing_page_url,
        'http://',
        ''
    )


,
        'https://',
        ''
    )



    )-coalesce(
            nullif(

    position(
        '/' in 

    replace(
        

    replace(
        landing_page_url,
        'http://',
        ''
    )


,
        'https://',
        ''
    )



    ), 0),
            

    position(
        '?' in 

    replace(
        

    replace(
        landing_page_url,
        'http://',
        ''
    )


,
        'https://',
        ''
    )



    ) - 1
            )
    ),
        '?',
        1
        )


  

 as TEXT)
 as url_path,
        nullif(

  
    

    split_part(
        

  
    

    split_part(
        landing_page_url,
        'utm_source=',
        2
        )


  

,
        '&',
        1
        )


  

,'') as utm_source,
        nullif(

  
    

    split_part(
        

  
    

    split_part(
        landing_page_url,
        'utm_medium=',
        2
        )


  

,
        '&',
        1
        )


  

,'') as utm_medium,
        nullif(

  
    

    split_part(
        

  
    

    split_part(
        landing_page_url,
        'utm_campaign=',
        2
        )


  

,
        '&',
        1
        )


  

,'') as utm_campaign,
        nullif(

  
    

    split_part(
        

  
    

    split_part(
        landing_page_url,
        'utm_content=',
        2
        )


  

,
        '&',
        1
        )


  

,'') as utm_content,
        nullif(

  
    

    split_part(
        

  
    

    split_part(
        landing_page_url,
        'utm_term=',
        2
        )


  

,
        '&',
        1
        )


  

,'') as utm_term,
        landing_page_url,
        row_number() over (partition by source_relation, ad_id order by updated_at desc) = 1 as is_most_recent_record
    from fields
)

select * 
from final