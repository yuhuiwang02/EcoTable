






    select
            "id",
  "_fivetran_synced",
  "allow_channelback",
  "assignee_id",
  "brand_id",
  "created_at",
  "description",
  "due_at",
  "external_id",
  "forum_topic_id",
  "group_id",
  "has_incidents",
  "is_public",
  "organization_id",
  "priority",
  "problem_id",
  "recipient",
  "requester_id",
  "status",
  "subject",
  "submitter_id",
  "system_client",
  "ticket_form_id",
  "type",
  "updated_at",
  "url",
  "via_channel",
  "via_source_from_id",
  "via_source_from_title",
  "via_source_rel",
  "via_source_to_address",
  "via_source_to_name",
  "merged_ticket_ids",
  "via_source_from_address",
  "followup_ids",
  "via_followup_source_id"
        from "google_ads"."public"."ticket_data" as source_table
    
    