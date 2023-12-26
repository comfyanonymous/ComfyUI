


import confluent_kafka

from config.config import CONFIG



class KafkaConnection:
    """
    """

    @staticmethod
    def create_producer():
        # connect
        conf = {
            'bootstrap.servers': CONFIG['kafka_settings']['url']
        }
        producer = confluent_kafka.Producer(conf)
            
        return producer
    
    
    @staticmethod
    def create_consumer(kfk_topic):
        # connect kafka
        kafka_config = CONFIG['kafka_settings']
        consumer_config = {
            'bootstrap.servers': kafka_config["url"],
            'group.id': kafka_config["group"]
            # 'auto.offset.reset': 'earliest'
        }
        consumer = confluent_kafka.Consumer(consumer_config)
        new_topic = confluent_kafka.cimpl.NewTopic(kfk_topic, 
                                                   num_partitions=kafka_config["topic_partitions"], 
                                                   replication_factor=kafka_config["topic_rep_factor"])
        try:
            from confluent_kafka.admin import AdminClient
            admin_client = AdminClient({'bootstrap.servers': kafka_config["url"]})
            admin_client.create_topics([new_topic])
            # AppLog.info(f'[Kafka] create topic: {kfk_topic}')
        except confluent_kafka.KafkaException as e:
            # AppLog.info(f'[Kafka] topic exist: {self.kfk_topic}')
            print(f'[Kafka] topic exist: {kfk_topic}')
            
        consumer.subscribe([kfk_topic])
        return consumer