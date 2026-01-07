with base as (

    select * 
    from "pendo"."public_stg_pendo"."stg_pendo__guide_history_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    app_id
    
 as 
    
    app_id
    
, 
    
    
    attribute_badge_can_change_badge_color
    
 as 
    
    attribute_badge_can_change_badge_color
    
, 
    
    
    attribute_badge_color
    
 as 
    
    attribute_badge_color
    
, 
    
    
    attribute_badge_height
    
 as 
    
    attribute_badge_height
    
, 
    
    
    attribute_badge_image_url
    
 as 
    
    attribute_badge_image_url
    
, 
    
    
    attribute_badge_is_only_show_once
    
 as 
    
    attribute_badge_is_only_show_once
    
, 
    
    
    attribute_badge_name
    
 as 
    
    attribute_badge_name
    
, 
    
    
    attribute_badge_offset_left
    
 as 
    
    attribute_badge_offset_left
    
, 
    
    
    attribute_badge_offset_right
    
 as 
    
    attribute_badge_offset_right
    
, 
    
    
    attribute_badge_offset_top
    
 as 
    
    attribute_badge_offset_top
    
, 
    
    
    attribute_badge_position
    
 as 
    
    attribute_badge_position
    
, 
    
    
    attribute_badge_show_on_event
    
 as 
    
    attribute_badge_show_on_event
    
, 
    
    
    attribute_badge_use_hover
    
 as 
    
    attribute_badge_use_hover
    
, 
    
    
    attribute_badge_width
    
 as 
    
    attribute_badge_width
    
, 
    
    
    attribute_device_type
    
 as 
    
    attribute_device_type
    
, 
    
    
    attribute_priority
    
 as 
    
    attribute_priority
    
, 
    
    
    attribute_type
    
 as 
    
    attribute_type
    
, 
    
    
    created_at
    
 as 
    
    created_at
    
, 
    
    
    created_by_user_id
    
 as 
    
    created_by_user_id
    
, 
    
    
    email_state
    
 as 
    
    email_state
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    is_multi_step
    
 as 
    
    is_multi_step
    
, 
    
    
    is_training
    
 as 
    
    is_training
    
, 
    
    
    last_updated_at
    
 as 
    
    last_updated_at
    
, 
    
    
    last_updated_by_user_id
    
 as 
    
    last_updated_by_user_id
    
, 
    
    
    launch_method
    
 as 
    
    launch_method
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    published_at
    
 as 
    
    published_at
    
, 
    
    
    recurrence
    
 as 
    
    recurrence
    
, 
    
    
    recurrence_eligibility_window
    
 as 
    
    recurrence_eligibility_window
    
, 
    
    
    reset_at
    
 as 
    
    reset_at
    
, 
    
    
    root_version_id
    
 as 
    
    root_version_id
    
, 
    
    
    stable_version_id
    
 as 
    
    stable_version_id
    
, 
    
    
    state
    
 as 
    
    state
    



        
    from base
),

final as (
    
    select 
        id as guide_id,
        name as guide_name,
        app_id,
        state,
        attribute_device_type as device_type,
        created_at,
        created_by_user_id,
        is_multi_step,
        is_training,
        last_updated_at,
        last_updated_by_user_id,
        launch_method,
        published_at,
        recurrence,
        recurrence_eligibility_window,
        reset_at,
        root_version_id,
        stable_version_id,
        
        _fivetran_synced

    from fields
)

select * 
from final