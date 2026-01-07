






    select
            "id",
  "_fivetran_synced",
  "body",
  "created",
  "facebook_comment",
  "public",
  "ticket_id",
  "tweet",
  "user_id",
  "voice_comment"
        from "google_ads"."public"."ticket_comment_data" as source_table
    
    