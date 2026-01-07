with base as (

    select * 
    from "instagram_business"."public_instagram_business_dev"."stg_instagram_business__media_insights_tmp"

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
    
    
    carousel_album_engagement
    
 as 
    
    carousel_album_engagement
    
, 
    
    
    carousel_album_reach
    
 as 
    
    carousel_album_reach
    
, 
    
    
    carousel_album_saved
    
 as 
    
    carousel_album_saved
    
, 
    
    
    carousel_album_shares
    
 as 
    
    carousel_album_shares
    
, 
    
    
    carousel_album_views
    
 as 
    
    carousel_album_views
    
, 
    
    
    comment_count
    
 as 
    
    comment_count
    
, 
    
    
    id
    
 as 
    
    id
    
, 
    
    
    like_count
    
 as 
    
    like_count
    
, 
    
    
    story_exits
    
 as 
    
    story_exits
    
, 
    
    
    story_reach
    
 as 
    
    story_reach
    
, 
    
    
    story_replies
    
 as 
    
    story_replies
    
, 
    
    
    story_taps_back
    
 as 
    
    story_taps_back
    
, 
    
    
    story_taps_forward
    
 as 
    
    story_taps_forward
    
, 
    
    
    story_shares
    
 as 
    
    story_shares
    
, 
    
    
    story_views
    
 as 
    
    story_views
    
, 
    
    
    video_photo_engagement
    
 as 
    
    video_photo_engagement
    
, 
    
    
    video_photo_reach
    
 as 
    
    video_photo_reach
    
, 
    
    
    video_photo_saved
    
 as 
    
    video_photo_saved
    
, 
    
    
    video_photo_shares
    
 as 
    
    video_photo_shares
    
, 
    
    
    video_photo_views
    
 as 
    
    video_photo_views
    
, 
    
    
    reel_comments
    
 as 
    
    reel_comments
    
, 
    
    
    reel_likes
    
 as 
    
    reel_likes
    
, 
    
    
    reel_reach
    
 as 
    
    reel_reach
    
, 
    
    
    reel_shares
    
 as 
    
    reel_shares
    
, 
    
    
    reel_total_interactions
    
 as 
    
    reel_total_interactions
    
, 
    
    
    reel_views
    
 as 
    
    reel_views
    
, 
    
    
    carousel_album_impressions
    
 as 
    
    carousel_album_impressions
    
, 
    
    
    carousel_album_video_views
    
 as 
    
    carousel_album_video_views
    
, 
    
    
    story_impressions
    
 as 
    
    story_impressions
    
, 
    
    
    video_photo_impressions
    
 as 
    
    video_photo_impressions
    
, 
    
    
    video_views
    
 as 
    
    video_views
    
, 
    
    
    reel_plays
    
 as 
    
    reel_plays
    




        


, cast('' as TEXT) as source_relation



        
    from base
),

final as (
    
select 
    _fivetran_id,
    _fivetran_synced,
    carousel_album_engagement,
    carousel_album_reach,
    carousel_album_saved,
    carousel_album_shares,
    carousel_album_views,
    comment_count,
    id as post_id,
    like_count,
    story_exits,
    story_reach,
    story_replies,
    story_taps_back,
    story_taps_forward,
    story_shares,
    story_views,
    video_photo_engagement,
    video_photo_reach,
    video_photo_saved,
    video_photo_shares,
    video_photo_views,
    reel_comments,
    reel_likes,
    reel_reach,
    reel_shares,
    reel_total_interactions,
    reel_views,
    source_relation,
    carousel_album_impressions, -- DEPRECATED
    carousel_album_video_views, -- DEPRECATED
    story_impressions, -- DEPRECATED
    video_photo_impressions, -- DEPRECATED
    video_views, -- DEPRECATED
    reel_plays -- DEPRECATED
from fields

),

is_most_recent as (

    select 
        *,
        row_number() over (partition by post_id, source_relation order by _fivetran_synced desc) = 1 as is_most_recent_record
    from final

)

select * from is_most_recent