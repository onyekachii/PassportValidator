using System;
using System.Collections.Generic;
using System.Text;

namespace Passport_Validator.Models
{
    internal class PassportValidationResult
    {
        public bool IsValid { get; set; }
        public string ErrorMessage { get; set; }
    }
}
