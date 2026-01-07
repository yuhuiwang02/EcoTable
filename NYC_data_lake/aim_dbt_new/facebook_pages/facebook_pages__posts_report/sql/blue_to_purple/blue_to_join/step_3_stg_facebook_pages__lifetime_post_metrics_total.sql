with base as (

    select * 
    from "facebook_pages"."public_stg_facebook_pages"."stg_facebook_pages__lifetime_post_metrics_total_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as timestamp) as 
    
    date
    
 , 
    cast(null as integer) as 
    
    post_clicks
    
 , 
    cast(null as integer) as 
    
    post_engaged_fan
    
 , 
    cast(null as integer) as 
    
    post_engaged_users
    
 , 
    cast(null as TEXT) as 
    
    post_id
    
 , 
    cast(null as integer) as 
    
    post_impressions
    
 , 
    cast(null as integer) as 
    
    post_impressions_fan
    
 , 
    cast(null as integer) as 
    
    post_impressions_nonviral
    
 , 
    cast(null as integer) as 
    
    post_impressions_organic
    
 , 
    cast(null as integer) as 
    
    post_impressions_paid
    
 , 
    cast(null as integer) as 
    
    post_impressions_viral
    
 , 
    cast(null as integer) as 
    
    post_negative_feedback
    
 , 
    cast(null as integer) as 
    
    post_reactions_anger_total
    
 , 
    cast(null as integer) as 
    
    post_reactions_haha_total
    
 , 
    cast(null as integer) as 
    
    post_reactions_like_total
    
 , 
    cast(null as integer) as 
    
    post_reactions_love_total
    
 , 
    cast(null as integer) as 
    
    post_reactions_sorry_total
    
 , 
    cast(null as integer) as 
    
    post_reactions_wow_total
    
 , 
    cast(null as integer) as 
    
    post_video_avg_time_watched
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_30_s_autoplayed
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_30_s_clicked_to_play
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_30_s_organic
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_30_s_paid
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_organic
    
 , 
    cast(null as integer) as 
    
    post_video_complete_views_paid
    
 , 
    cast(null as integer) as 
    
    post_video_length
    
 , 
    cast(null as integer) as 
    
    post_video_view_time
    
 , 
    cast(null as integer) as 
    
    post_video_view_time_organic
    
 , 
    cast(null as integer) as 
    
    post_video_views
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s_autoplayed
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s_clicked_to_play
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s_organic
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s_paid
    
 , 
    cast(null as integer) as 
    
    post_video_views_10_s_sound_on
    
 , 
    cast(null as integer) as 
    
    post_video_views_15_s
    
 , 
    cast(null as integer) as 
    
    post_video_views_autoplayed
    
 , 
    cast(null as integer) as 
    
    post_video_views_clicked_to_play
    
 , 
    cast(null as integer) as 
    
    post_video_views_organic
    
 , 
    cast(null as integer) as 
    
    post_video_views_paid
    
 , 
    cast(null as integer) as 
    
    post_video_views_sound_on
    
 


                
        


, cast('' as TEXT) as source_relation



        
    from base
),

final as (
    
    select
        _fivetran_synced,
        date as date_day,
        post_id,
        post_clicks as clicks,
        post_impressions as impressions,
        post_impressions_fan as impressions_fan,
        post_impressions_nonviral as impressions_nonviral,
        post_impressions_organic as impressions_organic,
        post_impressions_paid as impressions_paid,
        post_impressions_viral as impressions_viral,
        post_negative_feedback as negative_feedback,
        post_reactions_anger_total as reactions_anger_total,
        post_reactions_haha_total as reactions_haha_total,
        post_reactions_like_total as reactions_like_total,
        post_reactions_love_total as reactions_love_total,
        post_reactions_sorry_total as reactions_sorry_total,
        post_reactions_wow_total as reactions_wow_total,
        post_video_avg_time_watched / 1000.0 as video_avg_time_watched,
        post_video_complete_views_30_s_autoplayed as video_complete_views_30s_autoplayed,
        post_video_complete_views_30_s_clicked_to_play as video_complete_views_30s_clicked_to_play,
        post_video_complete_views_30_s_organic as video_complete_views_30s_organic,
        post_video_complete_views_30_s_paid as video_complete_views_30s_paid,
        post_video_complete_views_organic as video_complete_views_organic,
        post_video_complete_views_paid as video_complete_views_paid,
        post_video_length / 1000.0  as video_length,
        post_video_view_time / 1000.0 as video_view_time,
        post_video_view_time_organic / 1000.0 as video_view_time_organic,
        post_video_views as video_views,
        post_video_views_10_s as video_views_10s,
        post_video_views_10_s_autoplayed as video_views_10s_autoplayed,
        post_video_views_10_s_clicked_to_play as video_views_10s_clicked_to_play,
        post_video_views_10_s_organic as video_views_10s_organic,
        post_video_views_10_s_paid as video_views_10_s_paid,
        post_video_views_10_s_sound_on as video_views_10s_sound_on,
        post_video_views_15_s as video_views_15s,
        post_video_views_autoplayed as video_views_autoplayed,
        post_video_views_clicked_to_play as video_views_clicked_to_play,
        post_video_views_organic as video_views_organic,
        post_video_views_paid as video_views_paid,
        post_video_views_sound_on as video_views_sound_on,
        source_relation
    from fields
),

is_most_recent as (

    select 
        *,
        row_number() over (partition by post_id, source_relation order by date_day desc) = 1 as is_most_recent_record
    from final

)

select * from is_most_recent