from colab_train_llama32_remote import (
    hf_space_repo_to_base_url,
    make_training_args,
)


def test_hf_space_repo_to_base_url_formats_standard_hf_space_domain():
    assert (
        hf_space_repo_to_base_url("Ev3Dev/hackathon")
        == "https://ev3dev-hackathon.hf.space"
    )


def test_make_training_args_derives_base_url_when_missing():
    args = make_training_args(space_repo_id="Ev3Dev/hackathon", base_url="")
    assert args.base_url == "https://ev3dev-hackathon.hf.space"
    assert args.model_id == "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
