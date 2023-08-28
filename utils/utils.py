from datetime import datetime
from transformers import FlaxT5ForConditionalGeneration
from huggingface_hub import ModelCard, ModelCardData


def get_current_date():
    current_date = datetime.now().date()
    return current_date.strftime('%Y-%m-%d')


def generate_model_card(model_name='alzoubi36/t5-test'):
    card_data = ModelCardData(language='en', license='mit', library_name='keras')
    card = ModelCard.from_template(
        card_data,
        model_id=model_name,
        model_description="this model does this and that",
        use_auth_token='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'
    )
    card.push_to_hub()


def push_model_to_hub(path_to_model='t5-small', model_name='alzoubi36/t5-test'):
    FlaxT5ForConditionalGeneration.from_pretrained(path_to_model).push_to_hub(model_name,
                                                                              use_auth_token='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')


if __name__ == '__main__':
    # current_date_string = get_current_date()
    # push_model_to_hub()
    generate_model_card()