with base as (

    select * 
    from "facebook_pages"."public_stg_facebook_pages"."stg_facebook_pages__post_history_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    allowed_advertising_objects
    
 , 
    cast(null as timestamp) as 
    
    created_time
    
 , 
    cast(null as TEXT) as 
    
    id
    
 , 
    cast(null as boolean) as 
    
    is_eligible_for_promotion
    
 , 
    cast(null as boolean) as 
    
    is_hidden
    
 , 
    cast(null as boolean) as 
    
    is_instagram_eligible
    
 , 
    cast(null as boolean) as 
    
    is_published
    
 , 
    cast(null as TEXT) as 
    
    message
    
 , 
    cast(null as TEXT) as 
    
    page_id
    
 , 
    cast(null as TEXT) as 
    
    parent_id
    
 , 
    cast(null as TEXT) as 
    
    privacy_allow
    
 , 
    cast(null as TEXT) as 
    
    privacy_deny
    
 , 
    cast(null as TEXT) as 
    
    privacy_description
    
 , 
    cast(null as TEXT) as 
    
    privacy_friends
    
 , 
    cast(null as TEXT) as 
    
    privacy_value
    
 , 
    cast(null as TEXT) as 
    
    promotable_id
    
 , 
    cast(null as integer) as 
    
    share_count
    
 , 
    cast(null as TEXT) as 
    
    status_type
    
 , 
    cast(null as timestamp) as 
    
    updated_time
    
 


                
        


, cast('' as TEXT) as source_relation



        
    from base
),

final as (
    
    select
        _fivetran_synced,
        allowed_advertising_objects,
        created_time as created_timestamp,
        id as post_id,
        is_eligible_for_promotion,
        is_hidden,
        is_instagram_eligible,
        is_published,
        message as post_message,
        page_id,
        parent_id,
        privacy_allow,
        privacy_deny,
        privacy_description,
        privacy_friends,
        privacy_value,
        promotable_id,
        share_count,
        status_type,
        updated_time as updated_timestamp,
        'https://facebook.com/' || 

  
    

    split_part(
        id,
        '_',
        1
        )


  

 || '/posts/' || 

  
    

    split_part(
        id,
        '_',
        2
        )


  

 as post_url,
        source_relation
    from fields
)

select * 
from final