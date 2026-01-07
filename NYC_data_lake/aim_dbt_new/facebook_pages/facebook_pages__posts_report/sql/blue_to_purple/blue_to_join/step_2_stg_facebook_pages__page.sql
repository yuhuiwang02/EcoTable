with base as (

    select * 
    from "facebook_pages"."public_stg_facebook_pages"."stg_facebook_pages__page_tmp"

),

fields as (

    select
        
    cast(null as boolean) as 
    
    _fivetran_deleted
    
 , 
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    affiliation
    
 , 
    cast(null as TEXT) as 
    
    app_id
    
 , 
    cast(null as TEXT) as 
    
    artists_we_like
    
 , 
    cast(null as TEXT) as 
    
    attire
    
 , 
    cast(null as TEXT) as 
    
    awards
    
 , 
    cast(null as TEXT) as 
    
    band_interests
    
 , 
    cast(null as TEXT) as 
    
    band_members
    
 , 
    cast(null as TEXT) as 
    
    bio
    
 , 
    cast(null as TEXT) as 
    
    birthday
    
 , 
    cast(null as TEXT) as 
    
    booking_agent
    
 , 
    cast(null as TEXT) as 
    
    built
    
 , 
    cast(null as boolean) as 
    
    can_checkin
    
 , 
    cast(null as boolean) as 
    
    can_post
    
 , 
    cast(null as TEXT) as 
    
    category
    
 , 
    cast(null as TEXT) as 
    
    category_list
    
 , 
    cast(null as integer) as 
    
    checkins
    
 , 
    cast(null as TEXT) as 
    
    company_overview
    
 , 
    cast(null as TEXT) as 
    
    culinary_team
    
 , 
    cast(null as TEXT) as 
    
    current_location
    
 , 
    cast(null as TEXT) as 
    
    description
    
 , 
    cast(null as TEXT) as 
    
    directed_by
    
 , 
    cast(null as TEXT) as 
    
    display_subtext
    
 , 
    cast(null as TEXT) as 
    
    emails
    
 , 
    cast(null as integer) as 
    
    fan_count
    
 , 
    cast(null as TEXT) as 
    
    features
    
 , 
    cast(null as TEXT) as 
    
    food_styles
    
 , 
    cast(null as TEXT) as 
    
    founded
    
 , 
    cast(null as TEXT) as 
    
    general_info
    
 , 
    cast(null as TEXT) as 
    
    general_manager
    
 , 
    cast(null as TEXT) as 
    
    genre
    
 , 
    cast(null as TEXT) as 
    
    global_brand_page_name
    
 , 
    cast(null as boolean) as 
    
    has_added_app
    
 , 
    cast(null as boolean) as 
    
    has_transitioned_to_new_page_experience
    
 , 
    cast(null as boolean) as 
    
    has_whatsapp_number
    
 , 
    cast(null as TEXT) as 
    
    hometown
    
 , 
    cast(null as TEXT) as 
    
    id
    
 , 
    cast(null as TEXT) as 
    
    impressum
    
 , 
    cast(null as TEXT) as 
    
    influences
    
 , 
    cast(null as boolean) as 
    
    is_always_open
    
 , 
    cast(null as boolean) as 
    
    is_chain
    
 , 
    cast(null as boolean) as 
    
    is_community_page
    
 , 
    cast(null as boolean) as 
    
    is_eligible_for_branded_content
    
 , 
    cast(null as boolean) as 
    
    is_messenger_bot_get_started_enabled
    
 , 
    cast(null as boolean) as 
    
    is_messenger_platform_bot
    
 , 
    cast(null as boolean) as 
    
    is_owned
    
 , 
    cast(null as boolean) as 
    
    is_permanently_closed
    
 , 
    cast(null as boolean) as 
    
    is_published
    
 , 
    cast(null as boolean) as 
    
    is_unclaimed
    
 , 
    cast(null as TEXT) as 
    
    members
    
 , 
    cast(null as TEXT) as 
    
    mission
    
 , 
    cast(null as TEXT) as 
    
    mpg
    
 , 
    cast(null as TEXT) as 
    
    name
    
 , 
    cast(null as TEXT) as 
    
    network
    
 , 
    cast(null as integer) as 
    
    new_like_count
    
 , 
    cast(null as float) as 
    
    overall_star_rating
    
 , 
    cast(null as TEXT) as 
    
    personal_info
    
 , 
    cast(null as TEXT) as 
    
    personal_interests
    
 , 
    cast(null as TEXT) as 
    
    pharma_safety_info
    
 , 
    cast(null as TEXT) as 
    
    phone
    
 , 
    cast(null as TEXT) as 
    
    place_type
    
 , 
    cast(null as TEXT) as 
    
    plot_outline
    
 , 
    cast(null as TEXT) as 
    
    press_contact
    
 , 
    cast(null as TEXT) as 
    
    price_range
    
 , 
    cast(null as TEXT) as 
    
    produced_by
    
 , 
    cast(null as TEXT) as 
    
    products
    
 , 
    cast(null as boolean) as 
    
    promotion_eligible
    
 , 
    cast(null as TEXT) as 
    
    promotion_ineligible_reason
    
 , 
    cast(null as TEXT) as 
    
    public_transit
    
 , 
    cast(null as integer) as 
    
    rating_count
    
 , 
    cast(null as TEXT) as 
    
    record_label
    
 , 
    cast(null as TEXT) as 
    
    release_date
    
 , 
    cast(null as TEXT) as 
    
    schedule
    
 , 
    cast(null as TEXT) as 
    
    screenplay_by
    
 , 
    cast(null as TEXT) as 
    
    season
    
 , 
    cast(null as TEXT) as 
    
    single_line_address
    
 , 
    cast(null as TEXT) as 
    
    starring
    
 , 
    cast(null as integer) as 
    
    store_number
    
 , 
    cast(null as TEXT) as 
    
    studio
    
 , 
    cast(null as integer) as 
    
    talking_about_count
    
 , 
    cast(null as TEXT) as 
    
    username
    
 , 
    cast(null as TEXT) as 
    
    website
    
 , 
    cast(null as integer) as 
    
    were_here_count
    
 , 
    cast(null as TEXT) as 
    
    whatsapp_number
    
 , 
    cast(null as TEXT) as 
    
    written_by
    
 


                
        


, cast('' as TEXT) as source_relation



        
    from base
),

final as (
    
    select 
        _fivetran_deleted,
        _fivetran_synced,
        affiliation,
        app_id,
        artists_we_like,
        attire,
        awards,
        band_interests,
        band_members,
        bio,
        birthday,
        booking_agent,
        built,
        can_checkin,
        can_post,
        category,
        category_list,
        checkins,
        company_overview,
        culinary_team,
        current_location,
        description as page_description,
        directed_by,
        display_subtext,
        emails,
        fan_count,
        features,
        food_styles,
        founded,
        general_info,
        general_manager,
        genre,
        global_brand_page_name,
        has_added_app,
        has_transitioned_to_new_page_experience,
        has_whatsapp_number,
        hometown,
        id as page_id,
        impressum,
        influences,
        is_always_open,
        is_chain,
        is_community_page,
        is_eligible_for_branded_content,
        is_messenger_bot_get_started_enabled,
        is_messenger_platform_bot,
        is_owned,
        is_permanently_closed,
        is_published,
        is_unclaimed,
        members,
        mission,
        mpg,
        name as page_name,
        network,
        new_like_count,
        overall_star_rating,
        personal_info,
        personal_interests,
        pharma_safety_info,
        phone,
        place_type,
        plot_outline,
        press_contact,
        price_range,
        produced_by,
        products,
        promotion_eligible,
        promotion_ineligible_reason,
        public_transit,
        rating_count,
        record_label,
        release_date,
        schedule,
        screenplay_by,
        season,
        single_line_address,
        starring,
        store_number,
        studio,
        talking_about_count,
        username,
        website,
        were_here_count,
        whatsapp_number,
        written_by,
        source_relation
    from fields
)

select * from final