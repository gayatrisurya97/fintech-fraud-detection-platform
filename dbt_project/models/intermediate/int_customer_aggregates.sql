-- models/intermediate/int_customer_aggregates.sql
-- PURPOSE: Aggregate transaction data per customer
-- This tells us behavioural patterns for each customer
-- Used for customer segmentation in Power BI

with features as (

    select * from {{ ref('int_transaction_features') }}

),

aggregates as (

    select
        sender_id,

        -- Transaction behaviour
        count(*)                                    as total_transactions,
        round(sum(amount), 2)                       as total_spend,
        round(avg(amount), 2)                       as avg_transaction_amount,
        round(max(amount), 2)                       as max_transaction_amount,
        round(min(amount), 2)                       as min_transaction_amount,

        -- Fraud behaviour
        sum(is_fraud)                               as total_fraud_transactions,
        round(avg(is_fraud) * 100, 2)               as fraud_rate_percent,

        -- Balance behaviour
        round(avg(sender_balance_before), 2)        as avg_balance,
        sum(sender_balance_zero)                    as times_balance_hit_zero,

        -- Transaction types
        count(case when transaction_type = 'TRANSFER' then 1 end)   as transfer_count,
        count(case when transaction_type = 'CASH_OUT' then 1 end)   as cashout_count,
        count(case when transaction_type = 'PAYMENT' then 1 end)    as payment_count,
        count(case when transaction_type = 'CASH_IN' then 1 end)    as cashin_count,

        -- Customer segment based on total spend
        case
            when sum(amount) >= 500000  then 'High Value'
            when sum(amount) >= 100000  then 'Medium Value'
            when sum(amount) >= 10000   then 'Low Value'
            else 'Dormant'
        end as customer_segment

    from features

    group by sender_id

)

select * from aggregates