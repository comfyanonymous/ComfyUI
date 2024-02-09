from typing import TypedDict

import jwt


class ComfyJwt(TypedDict, total=False):
    sub: str


def jwt_decode(user_token: str) -> ComfyJwt:
    # todo: set up a way for users to override this behavior easily
    return ComfyJwt(**jwt.decode(user_token, algorithms=['HS256', "none"],
                                 # todo: this should be configurable
                                 options={"verify_signature": False, 'verify_aud': False, 'verify_iss': False}))
