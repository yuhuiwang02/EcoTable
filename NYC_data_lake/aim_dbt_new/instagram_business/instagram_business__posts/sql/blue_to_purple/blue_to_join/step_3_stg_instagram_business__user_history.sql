with base as (

    select * 
    from "instagram_business"."public_instagram_business_dev"."stg_instagram_business__user_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_id
    
 as 
    
    _fivetran_id
    
, 
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    followers_count
    
 as 
    
    followers_count
    
, 
    
    
    follows_count
    
 as 
    
    follows_count
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    ig_id
    
 as 
    
    ig_id
    
, 
    
    
    media_count
    
 as 
    
    media_count
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    username
    
 as 
    
    username
    
, 
    
    
    website
    
 as 
    
    website
    




        


, cast('' as TEXT) as source_relation



        
    from base
),

final as (
    
    select 
        _fivetran_id,
        _fivetran_synced,
        followers_count,
        follows_count,
        id as user_id,
        ig_id,
        media_count,
        name as account_name,
        username,
        website,
        source_relation
    from fields
),

is_most_recent as (

    select 
        *,
        row_number() over (partition by user_id, source_relation order by _fivetran_synced desc) = 1 as is_most_recent_record
    from final

)

select * from is_most_recent