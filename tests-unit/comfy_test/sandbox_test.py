from sandbox import windows_sandbox

def test_icacl_no_low_integrity_label():
    icacl_output = r"""
    foo NT AUTHORITY\SYSTEM:(OI)(CI)(F)
    """
    assert not windows_sandbox.does_permit_low_integrity_write(icacl_output)

def test_icacl_missing_inherit_flags():
    icacl_output = r"""
    foo Mandatory Label\Low Mandatory Level:(NW)
    """
    assert not windows_sandbox.does_permit_low_integrity_write(icacl_output)

    icacl_output = r"""
    foo Mandatory Label\Low Mandatory Level:(OI)(NW)
    """
    assert not windows_sandbox.does_permit_low_integrity_write(icacl_output)

    icacl_output = r"""
    foo Mandatory Label\Low Mandatory Level:(CI)(NW)
    """
    assert not windows_sandbox.does_permit_low_integrity_write(icacl_output)

def test_icacl_correct_acls():
    icacl_output = r"""
    foo Mandatory Label\Low Mandatory Level:(I)(OI)(CI)(NW)
    """
    assert windows_sandbox.does_permit_low_integrity_write(icacl_output)

    icacl_output = r"""
    foo Mandatory Label\Low Mandatory Level:(OI)(CI)(NW)
    """
    assert windows_sandbox.does_permit_low_integrity_write(icacl_output)
