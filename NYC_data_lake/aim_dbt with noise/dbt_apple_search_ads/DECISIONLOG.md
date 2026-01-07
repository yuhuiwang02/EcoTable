# Decision Log

## Conversion Value
There is no direct `conversion_value` field available in Apple Search Ads data. Alternatively, users may be able to leverage the `sales_purchasable_item_source_type_report` table from the Apple App Store connector, which tracks app and in-app purchases. The `sales` field, when filtered by `source_type = 'App Store Search'`, provides insights into purchases originating from App Store searches. See [Apple's Sales and Trends metrics and dimensions Doc](https://developer.apple.com/help/app-store-connect/reference/sales-and-trends-metrics-and-dimensions/) for more details.

Currently, our Apple App Store package does not utilize the `sales_purchasable_item_source_type_report` table. Go to this [Github Issue](https://github.com/fivetran/dbt_apple_store/issues/27) section to join the discussion.
